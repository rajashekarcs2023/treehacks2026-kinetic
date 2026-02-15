"""
AEGIS Motion Generation — Modal GPU Deployment

Deploys MoMask (CVPR 2024) on Modal's A100 GPU for text-to-3D skeleton generation.

Deploy:   modal deploy dgx/modal_motion.py
Test:     modal run dgx/modal_motion.py --prompt "a person doing a squat"
Endpoint: https://<your-username>--aegis-motion-generate.modal.run

The endpoint accepts POST with JSON: {"prompt": "...", "num_frames": 60}
Returns: {"keypoints": [...], "num_frames": 60, ...}
"""

import modal

# ── Modal App ────────────────────────────────────────────────
app = modal.App("aegis-motion")

# ── Build the image with all deps ────────────────────────────
momask_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install(
        "torch==2.1.0",
        "numpy",
        "scipy",
        "tqdm",
        "transformers",
        "clip@git+https://github.com/openai/CLIP.git",
        "gdown",
        "einops",
        "smplx",
        "trimesh",
    )
    .run_commands(
        "git clone https://github.com/EricGuo5513/momask-codes.git /momask",
        "cd /momask && mkdir -p checkpoints/t2m",
        # Download pre-trained MoMask weights
        "gdown 1VzVUbTqdELGsfXBnBjlqCBYSD7vVJfKQ -O /momask/checkpoints/t2m/momask.tar.gz || true",
        "cd /momask/checkpoints/t2m && tar -xzf momask.tar.gz 2>/dev/null || true",
        # Download T2M evaluator (needed for text encoding)
        "cd /momask && python -m spacy download en_core_web_sm 2>/dev/null || true",
        # Download CLIP
        "python -c \"import clip; clip.load('ViT-B/32', device='cpu')\" || true",
    )
)

# ── SMPL joint names (22 joints) ────────────────────────────
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine", "left_knee", "right_knee",
    "spine1", "left_ankle", "right_ankle", "spine2", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

# Map SMPL 22 → MediaPipe 33 (approximate)
SMPL_TO_MP = {
    0: 23, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28,
    10: 31, 11: 32, 15: 0, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16,
}


def joints_to_mediapipe(joints_3d, frame_w=640, frame_h=480):
    """Convert SMPL 22 joints (3D) to MediaPipe 33-point 2D format."""
    import numpy as np
    frames = []
    for frame in joints_3d:
        mp33 = [{"x": frame_w / 2, "y": frame_h / 2, "vis": 0.5}] * 33
        for smpl_idx, mp_idx in SMPL_TO_MP.items():
            if smpl_idx < len(frame):
                j = frame[smpl_idx]
                x = float(j[0]) * frame_w * 0.3 + frame_w / 2
                y = float(-j[1]) * frame_h * 0.3 + frame_h / 2
                mp33[mp_idx] = {"x": round(x, 2), "y": round(y, 2), "vis": 0.95}

        # Fill face points from head
        if mp33[0]["vis"] > 0.5:
            nx, ny = mp33[0]["x"], mp33[0]["y"]
            for i, (dx, dy) in enumerate([
                (-12, -8), (-8, -10), (12, -8), (8, -10),
                (-16, -6), (16, -6), (-25, 0), (25, 0), (-5, 15), (5, 15)
            ]):
                mp33[i + 1] = {"x": nx + dx, "y": ny + dy, "vis": 0.9}

        # Fill hands from wrists
        for w_idx, base in [(15, 17), (16, 18)]:
            if mp33[w_idx]["vis"] > 0.5:
                wx, wy = mp33[w_idx]["x"], mp33[w_idx]["y"]
                s = -1 if w_idx == 15 else 1
                for k, (dx, dy) in enumerate([(s*5, 8), (s*3, 12), (-s*5, 5), (s*4, 10)]):
                    if base + k < 33:
                        mp33[base + k] = {"x": wx + dx, "y": wy + dy, "vis": 0.9}

        # Feet from ankles
        for a_idx, f_idx, h_idx in [(27, 31, 29), (28, 32, 30)]:
            if mp33[a_idx]["vis"] > 0.5:
                ax, ay = mp33[a_idx]["x"], mp33[a_idx]["y"]
                s = -1 if a_idx == 27 else 1
                mp33[f_idx] = {"x": ax + s * 8, "y": ay + 10, "vis": 0.9}
                mp33[h_idx] = {"x": ax - s * 5, "y": ay + 5, "vis": 0.9}

        frames.append(mp33)
    return frames


@app.function(
    image=momask_image,
    gpu="A100",
    timeout=120,
    memory=16384,
)
def generate(prompt: str, num_frames: int = 60):
    """Generate 3D skeleton motion from text using MoMask on A100 GPU."""
    import sys
    import time
    import numpy as np
    import torch

    sys.path.insert(0, "/momask")
    t0 = time.time()

    try:
        # Try using MoMask's generation interface
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "gen_t2m.py",
                "--text_prompt", prompt,
                "--lengths", str(num_frames),
                "--num_repetitions", "1",
                "--num_samples", "1",
            ],
            cwd="/momask",
            capture_output=True, text=True, timeout=90,
        )

        # Find output
        import glob
        npy_files = sorted(glob.glob("/momask/generation/**/*.npy", recursive=True))
        if npy_files:
            joints = np.load(npy_files[-1])  # (num_frames, 22, 3)
            gen_ms = (time.time() - t0) * 1000

            mp_keypoints = joints_to_mediapipe(joints.tolist())

            return {
                "prompt": prompt,
                "num_frames": len(mp_keypoints),
                "model": "MoMask (CVPR 2024)",
                "device": "NVIDIA A100 (Modal)",
                "generation_ms": round(gen_ms, 1),
                "keypoints": mp_keypoints,
                "raw_joints_shape": list(joints.shape),
            }
        else:
            return {"error": f"No output generated. stdout: {result.stdout[:300]}, stderr: {result.stderr[:300]}"}

    except Exception as e:
        return {"error": str(e)}


@app.function(
    image=momask_image,
    gpu="A100",
    timeout=120,
    memory=16384,
)
@modal.web_endpoint(method="POST")
def generate_endpoint(item: dict):
    """Web endpoint for motion generation. POST {"prompt": "...", "num_frames": 60}"""
    prompt = item.get("prompt", "a person walking forward")
    num_frames = item.get("num_frames", 60)
    return generate.remote(prompt, num_frames)


# ── CLI test ─────────────────────────────────────────────────
@app.local_entrypoint()
def main(prompt: str = "a person doing a squat"):
    print(f"Generating motion for: '{prompt}'")
    result = generate.remote(prompt)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Generated {result['num_frames']} frames in {result['generation_ms']}ms")
        print(f"Model: {result['model']} on {result['device']}")
