"""
AEGIS Motion Generation — HY-Motion 1.0 on Modal A100 GPU

Deploys Tencent's HY-Motion 1.0 (SOTA text-to-3D motion, Dec 2025)
on Modal's A100 GPU for text-to-skeleton generation.

Setup:    python3 -m modal setup   (one-time browser auth)
Deploy:   modal deploy dgx/modal_motion.py
Test:     modal run dgx/modal_motion.py --prompt "a person doing a squat"
Endpoint: https://<your-workspace>--aegis-motion-generate.modal.run
"""

import os
import time
import modal

# ── App + Volume for caching model weights ───────────────────
app = modal.App("aegis-motion")
hf_cache_vol = modal.Volume.from_name("aegis-hf-cache", create_if_missing=True)

# ── Download HY-Motion weights into the repo's ckpts dir ─────
def download_hymotion():
    """Download HY-Motion 1.0-Lite weights from HuggingFace into ckpts/tencent/."""
    import subprocess
    # Weights must be at /hymotion/ckpts/tencent/HY-Motion-1.0-Lite
    ckpt_dir = "/hymotion/ckpts/tencent"
    subprocess.run(["mkdir", "-p", ckpt_dir], check=True)

    from huggingface_hub import snapshot_download

    # Download HY-Motion 1.0-Lite model weights
    snapshot_download(
        repo_id="tencent/HY-Motion-1.0",
        local_dir="/hymotion/ckpts/tencent",
        allow_patterns=["HY-Motion-1.0-Lite/**"],
    )
    print("[KINETIC] HY-Motion 1.0-Lite weights downloaded to /hymotion/ckpts/tencent/")

    # Download CLIP text encoder (required by HY-Motion for text encoding)
    snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir="/hymotion/ckpts/clip-vit-large-patch14",
    )
    print("[KINETIC] CLIP text encoder downloaded")

    # Download Qwen3-8B LLM text encoder (required by HY-Motion config: llm_type=qwen3)
    snapshot_download(
        repo_id="Qwen/Qwen3-8B",
        local_dir="/hymotion/ckpts/Qwen3-8B",
    )
    print("[KINETIC] Qwen3-8B text encoder downloaded")

    # Verify
    import os
    lite_dir = os.path.join(ckpt_dir, "HY-Motion-1.0-Lite")
    if os.path.exists(lite_dir):
        files = os.listdir(lite_dir)
        print(f"[AEGIS] Found {len(files)} files in HY-Motion-1.0-Lite: {files[:5]}")
    else:
        print(f"[AEGIS] WARNING: {lite_dir} not found after download!")


# ── Build the container image ────────────────────────────────
hymotion_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "ffmpeg")
    # Clone repo first
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git /hymotion",
        "cd /hymotion && git lfs install && git lfs pull",
    )
    # Install exact deps from HY-Motion requirements.txt
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
    )
    .pip_install("fastapi[standard]")
    .pip_install(
        "huggingface_hub==0.30.0",
        "torchdiffeq==0.2.5",
        "accelerate==0.30.1",
        "diffusers==0.26.3",
        "transformers==4.53.3",
        "einops==0.8.1",
        "safetensors==0.5.3",
        "bitsandbytes==0.49.0",
        "numpy<2.0",
        "scipy>=1.10.0",
        "transforms3d==0.4.2",
        "PyYAML==6.0",
        "omegaconf==2.3.0",
        "click==8.1.3",
        "requests==2.32.4",
        "openai>=1.0.0",
    )
    # fbxsdkpy is optional (only for FBX export), skip if it fails
    .run_commands(
        "pip install --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple fbxsdkpy==2020.1.post2 2>/dev/null || echo 'fbxsdkpy skipped (optional, FBX export only)'",
    )
    # Download model weights (needs HF token for gated models)
    .run_function(
        download_hymotion,
        volumes={"/root/.cache/huggingface": hf_cache_vol},
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)


# ── SMPL 22-joint → MediaPipe 33-point mapping ──────────────
SMPL_TO_MP = {
    0: 23, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28,
    10: 31, 11: 32, 15: 0, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16,
}


def joints_to_mediapipe(joints_3d, frame_w=640, frame_h=480):
    """Convert SMPL 22 joints (3D) → MediaPipe 33-point 2D format."""
    frames = []
    for frame in joints_3d:
        mp33 = [{"x": frame_w / 2, "y": frame_h / 2, "vis": 0.5}] * 33
        for smpl_idx, mp_idx in SMPL_TO_MP.items():
            if smpl_idx < len(frame):
                j = frame[smpl_idx]
                x = float(j[0]) * frame_w * 0.3 + frame_w / 2
                y = float(-j[1]) * frame_h * 0.3 + frame_h / 2
                mp33[mp_idx] = {"x": round(x, 2), "y": round(y, 2), "vis": 0.95}

        # Fill face from head
        if mp33[0]["vis"] > 0.5:
            nx, ny = mp33[0]["x"], mp33[0]["y"]
            offsets = [(-12,-8),(-8,-10),(12,-8),(8,-10),(-16,-6),(16,-6),(-25,0),(25,0),(-5,15),(5,15)]
            for i, (dx, dy) in enumerate(offsets):
                mp33[i + 1] = {"x": nx + dx, "y": ny + dy, "vis": 0.9}

        # Fill hands from wrists
        for w_idx, base in [(15, 17), (16, 18)]:
            if mp33[w_idx]["vis"] > 0.5:
                wx, wy = mp33[w_idx]["x"], mp33[w_idx]["y"]
                s = -1 if w_idx == 15 else 1
                for k, (dx, dy) in enumerate([(s*5,8),(s*3,12),(-s*5,5),(s*4,10)]):
                    if base + k < 33:
                        mp33[base + k] = {"x": wx+dx, "y": wy+dy, "vis": 0.9}

        # Feet from ankles
        for a_idx, f_idx, h_idx in [(27,31,29),(28,32,30)]:
            if mp33[a_idx]["vis"] > 0.5:
                ax, ay = mp33[a_idx]["x"], mp33[a_idx]["y"]
                s = -1 if a_idx == 27 else 1
                mp33[f_idx] = {"x": ax+s*8, "y": ay+10, "vis": 0.9}
                mp33[h_idx] = {"x": ax-s*5, "y": ay+5, "vis": 0.9}

        frames.append(mp33)
    return frames


# ── Cached runtime (loaded once per container) ───────────────
_runtime = None

def _get_runtime():
    """Load T2MRuntime once, reuse across requests."""
    global _runtime
    if _runtime is not None:
        return _runtime

    import sys
    sys.path.insert(0, "/hymotion")
    os.chdir("/hymotion")  # T2MRuntime uses relative paths for body model assets

    from hymotion.utils.t2m_runtime import T2MRuntime

    model_dir = "/hymotion/ckpts/tencent/HY-Motion-1.0-Lite"
    config_path = os.path.join(model_dir, "config.yml")
    ckpt_path = os.path.join(model_dir, "latest.ckpt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"HY-Motion config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HY-Motion checkpoint not found: {ckpt_path}")

    print(f"[AEGIS] Loading T2MRuntime from {model_dir}...")
    _runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        device_ids=[0],
        disable_prompt_engineering=True,
    )
    print("[AEGIS] T2MRuntime loaded ✓")
    return _runtime


# ── Main generation function ─────────────────────────────────
@app.function(
    image=hymotion_image,
    gpu="A100",
    timeout=180,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(prompt: str, num_frames: int = 60):
    """Generate 3D motion from text using HY-Motion 1.0 on A100 GPU."""
    import glob
    import numpy as np

    t0 = time.time()
    output_dir = "/tmp/motion_output"
    os.makedirs(output_dir, exist_ok=True)

    runtime = _get_runtime()

    # Duration in seconds (num_frames at 30fps)
    duration = max(1.0, num_frames / 30.0)

    # Generate motion using Python API directly
    html, saved_files, motion_data = runtime.generate_motion(
        text=prompt,
        seeds_csv="42",
        duration=duration,
        cfg_scale=5.0,
        output_format="dict",  # dict format when fbxsdkpy unavailable
        original_text=prompt,
        output_dir=output_dir,
        output_filename="aegis_gen",
    )

    gen_ms = (time.time() - t0) * 1000

    # Try to extract joint positions from motion_data
    # HY-Motion returns joint positions in SMPL format
    joints = None

    # Check if motion_data contains joint positions directly
    if motion_data is not None:
        if isinstance(motion_data, dict):
            # Look for joint data in common keys
            for key in ["joints", "motion", "positions", "joint_positions", "pred_motion"]:
                if key in motion_data:
                    data = motion_data[key]
                    if hasattr(data, 'cpu'):
                        data = data.cpu().numpy()
                    joints = np.array(data)
                    break
            if joints is None:
                # Dump keys for debugging
                return {
                    "error": "Could not find joint data in motion_data dict",
                    "available_keys": list(motion_data.keys())[:20],
                    "generation_ms": round(gen_ms, 1),
                }
        elif hasattr(motion_data, 'cpu'):
            joints = motion_data.cpu().numpy()
        elif isinstance(motion_data, np.ndarray):
            joints = motion_data

    # Also check for any .npy files saved to output_dir
    if joints is None:
        npy_files = sorted(glob.glob(os.path.join(output_dir, "**/*.npy"), recursive=True))
        if npy_files:
            joints = np.load(npy_files[-1])

    if joints is not None:
        # Reshape if needed: expect (num_frames, num_joints, 3)
        if joints.ndim == 2:
            # Might be (num_frames, num_joints*3)
            joints = joints.reshape(joints.shape[0], -1, 3)

        mp_keypoints = joints_to_mediapipe(joints.tolist())

        return {
            "prompt": prompt,
            "num_frames": len(mp_keypoints),
            "model": "HY-Motion 1.0-Lite (Tencent, SOTA Dec 2025)",
            "device": "NVIDIA A100 (Modal)",
            "generation_ms": round(gen_ms, 1),
            "keypoints": mp_keypoints,
            "raw_shape": list(joints.shape),
        }

    # If we still have no joints, return what we have
    all_files = sorted(glob.glob(os.path.join(output_dir, "**/*"), recursive=True))
    return {
        "error": "Motion generated but could not extract joint positions",
        "saved_files": [os.path.basename(f) for f in saved_files] if saved_files else [],
        "output_files": [os.path.basename(f) for f in all_files[:10]],
        "generation_ms": round(gen_ms, 1),
        "motion_data_type": str(type(motion_data)),
    }


# ── Web endpoint (POST) — no GPU needed, proxies to generate() ──
@app.function(
    image=hymotion_image,
    timeout=180,
)
@modal.fastapi_endpoint(method="POST")
def generate_endpoint(item: dict):
    """POST {"prompt": "a person doing a squat", "num_frames": 60}"""
    prompt = item.get("prompt", "a person walking forward")
    num_frames = item.get("num_frames", 60)
    return generate.remote(prompt, num_frames)


# ── CLI test ─────────────────────────────────────────────────
@app.local_entrypoint()
def main(prompt: str = "a person doing a squat"):
    print(f"Generating motion for: '{prompt}'")
    print(f"Using HY-Motion 1.0-Lite on Modal A100 GPU...")
    result = generate.remote(prompt)
    if "error" in result:
        print(f"Error: {result['error']}")
        if "stderr" in result:
            print(f"stderr: {result['stderr']}")
    else:
        print(f"✓ Generated {result['num_frames']} frames in {result['generation_ms']:.0f}ms")
        print(f"  Model: {result['model']}")
        print(f"  Device: {result['device']}")
        print(f"  Shape: {result['raw_shape']}")
