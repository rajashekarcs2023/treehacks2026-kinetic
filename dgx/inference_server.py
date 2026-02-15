"""
AEGIS DGX Inference Server — Whole-Body Pose Estimation on NVIDIA DGX Spark.

Uses YOLOv8n-pose for 17-keypoint body pose + ultralytics framework,
running on NVIDIA DGX Spark's Grace ARM CPU (aarch64, 20 cores).

Usage:
    python inference_server.py [--port 8080]

Architecture:
    AEGIS (laptop) → POST /predict (image) → DGX Spark (this server) → keypoints
    Fallback: If DGX unreachable, AEGIS uses local MediaPipe (33 body + 21/hand)
"""

import argparse
import io
import os
import time
import platform

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="AEGIS DGX Inference", version="2.0")

# ── Global models ────────────────────────────────────────────
pose_model = None
motion_model = None          # MDM / MLD motion generation
motion_model_type = None     # "mdm" | "mld" | None
request_count = 0
total_inference_ms = 0
motion_request_count = 0

BODY_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_models():
    """Load YOLOv8-pose model via ultralytics."""
    global pose_model
    try:
        import torch
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # YOLOv8n-pose: 17 keypoints, fast, well-tested
        pose_model = YOLO("yolov8n-pose.pt")
        pose_model.to(device)
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
        print(f"[DGX] YOLOv8n-pose loaded (17 body keypoints, {device} mode)")
        print(f"[DGX] Platform: {platform.machine()} ({os.cpu_count()} cores)")
        if device == "cuda":
            print(f"[DGX] GPU: {gpu_name}")

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        pose_model(dummy, verbose=False)
        print("[DGX] Warmup complete")
    except Exception as e:
        print(f"[DGX] ERROR loading model: {e}")

    # Try loading motion generation model
    load_motion_model()


def load_motion_model():
    """Try loading a text-to-motion generation model. Non-blocking — skips if unavailable."""
    global motion_model, motion_model_type

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DGX Motion] Device: {device}")
    if device == "cuda":
        print(f"[DGX Motion] GPU: {torch.cuda.get_device_name(0)}")

    # Priority 1: HY-Motion 1.0 (Tencent, SOTA, needs GPU)
    hymotion_dir = os.environ.get("HYMOTION_DIR", os.path.expanduser("~/aegis-dgx/HY-Motion-1.0"))
    if os.path.exists(hymotion_dir):
        try:
            import sys
            if hymotion_dir not in sys.path:
                sys.path.insert(0, hymotion_dir)
            motion_model = {"type": "hymotion", "dir": hymotion_dir, "device": device}
            motion_model_type = "hymotion"
            print(f"[DGX Motion] HY-Motion 1.0 found at {hymotion_dir} — SOTA text-to-motion on {device}")
            return
        except Exception as e:
            print(f"[DGX Motion] HY-Motion 1.0 load error: {e}")

    # Priority 2: MoMask (CVPR 2024, good quality)
    momask_dir = os.environ.get("MOMASK_DIR", os.path.expanduser("~/aegis-dgx/MoMask"))
    if os.path.exists(momask_dir):
        try:
            motion_model = {"type": "momask", "dir": momask_dir, "device": device}
            motion_model_type = "momask"
            print(f"[DGX Motion] MoMask found at {momask_dir}")
            return
        except Exception as e:
            print(f"[DGX Motion] MoMask load error: {e}")

    # Priority 3: MLD (fast, lighter)
    mld_ckpt = os.environ.get("MLD_CHECKPOINT", "models/mld_humanml3d.ckpt")
    if os.path.exists(mld_ckpt):
        motion_model = {"type": "mld", "checkpoint": mld_ckpt, "device": device}
        motion_model_type = "mld"
        print(f"[DGX Motion] MLD model ready on {device}")
        return

    # Priority 4: Modal endpoint (remote GPU — A100)
    modal_url = os.environ.get("MODAL_MOTION_URL", "")
    if modal_url:
        motion_model = {"type": "modal", "url": modal_url}
        motion_model_type = "modal"
        print(f"[DGX Motion] Modal endpoint configured: {modal_url}")
        return

    print("[DGX Motion] No motion generation model found — /generate_motion will return 503")
    print("[DGX Motion] Options:")
    print("[DGX Motion]   1. Deploy Modal: modal deploy dgx/modal_motion.py")
    print("[DGX Motion]   2. Set MODAL_MOTION_URL=https://your--aegis-motion-generate-endpoint.modal.run")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "pose_model_loaded": pose_model is not None,
        "motion_model_loaded": motion_model is not None,
        "motion_model_type": motion_model_type,
        "model": "YOLOv8n-pose (17 body keypoints)",
        "platform": f"{platform.machine()} ({os.cpu_count()} cores)",
        "device": "NVIDIA DGX Spark (Grace ARM CPU)",
        "total_pose_requests": request_count,
        "total_motion_requests": motion_request_count,
        "avg_inference_ms": round(total_inference_ms / max(request_count, 1), 1),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run pose estimation on an uploaded image.

    Returns 17 body keypoints with detection confidence.
    Each keypoint has: index, name, x (0-1), y (0-1), confidence.
    """
    global request_count, total_inference_ms

    if pose_model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Pose model not loaded"}
        )

    t0 = time.time()

    # Read image
    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    h, w = img.shape[:2]

    # Run YOLOv8-pose inference
    results = pose_model(img, verbose=False)

    inference_ms = (time.time() - t0) * 1000
    request_count += 1
    total_inference_ms += inference_ms

    if not results or len(results) == 0:
        return JSONResponse(
            status_code=422,
            content={"error": "No pose detected in image"}
        )

    result = results[0]

    # Extract keypoints from best detection
    persons = []
    if result.keypoints is not None and len(result.keypoints) > 0:
        for person_idx in range(len(result.keypoints)):
            kpts = result.keypoints[person_idx]
            keypoints = []

            if kpts.xy is not None and len(kpts.xy) > 0:
                xy = kpts.xy[0] if len(kpts.xy.shape) == 3 else kpts.xy
                conf_data = kpts.conf[0] if kpts.conf is not None and len(kpts.conf.shape) == 2 else kpts.conf

                for k in range(min(len(xy), 17)):
                    kpt_x = float(xy[k][0])
                    kpt_y = float(xy[k][1])
                    kpt_conf = float(conf_data[k]) if conf_data is not None and k < len(conf_data) else 0.0

                    keypoints.append({
                        "index": k,
                        "name": BODY_NAMES[k] if k < len(BODY_NAMES) else f"kpt_{k}",
                        "x": round(kpt_x / w, 4),
                        "y": round(kpt_y / h, 4),
                        "confidence": round(kpt_conf, 3),
                    })

            # Get bounding box
            bbox = None
            if result.boxes is not None and person_idx < len(result.boxes):
                box = result.boxes[person_idx]
                bbox = {
                    "x1": round(float(box.xyxy[0][0]) / w, 4),
                    "y1": round(float(box.xyxy[0][1]) / h, 4),
                    "x2": round(float(box.xyxy[0][2]) / w, 4),
                    "y2": round(float(box.xyxy[0][3]) / h, 4),
                    "confidence": round(float(box.conf[0]), 3),
                }

            persons.append({
                "person_index": person_idx,
                "bbox": bbox,
                "keypoints": keypoints,
                "num_keypoints": len(keypoints),
            })

    return {
        "persons_detected": len(persons),
        "model": "YOLOv8n-pose",
        "device": "NVIDIA DGX Spark (Grace CPU)",
        "inference_ms": round(inference_ms, 1),
        "image_size": {"width": w, "height": h},
        "persons": persons,
    }


# ── Motion Generation ────────────────────────────────────────

# SMPL 22-joint to MediaPipe 33-point mapping (approximate)
SMPL_TO_MP33 = {
    0: 23,   # pelvis → left_hip (approx midpoint)
    1: 23,   # left_hip
    2: 24,   # right_hip
    3: 23,   # spine (→ left_hip as proxy)
    4: 25,   # left_knee
    5: 26,   # right_knee
    6: 23,   # spine1
    7: 27,   # left_ankle
    8: 28,   # right_ankle
    9: 31,   # left_foot
    10: 32,  # right_foot
    12: 0,   # head → nose
    13: 11,  # left_collar → left_shoulder
    14: 12,  # right_collar → right_shoulder
    15: 13,  # left_elbow
    16: 14,  # right_elbow
    17: 15,  # left_wrist
    18: 16,  # right_wrist
}


async def _run_modal_inference(prompt: str, num_frames: int = 60):
    """Proxy motion generation to Modal's A100 GPU endpoint."""
    import httpx

    modal_url = motion_model["url"]
    t0 = time.time()

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            modal_url,
            json={"prompt": prompt, "num_frames": num_frames},
        )

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={"error": f"Modal error: {resp.text[:300]}"})

    data = resp.json()
    data["proxy"] = "DGX Spark → Modal A100"
    data["total_ms"] = round((time.time() - t0) * 1000, 1)

    global motion_request_count
    motion_request_count += 1

    return data


def _run_hymotion_inference(prompt: str, num_frames: int = 60) -> list:
    """Run HY-Motion 1.0 inference on GPU. Returns list of frames with 22 SMPL joints (x,y,z)."""
    import sys
    import torch
    import subprocess
    import json as json_mod

    hymotion_dir = motion_model["dir"]

    # Write prompt to temp file
    prompt_file = os.path.join(hymotion_dir, "temp_prompt.txt")
    output_dir = os.path.join(hymotion_dir, "output", "api_output")
    os.makedirs(output_dir, exist_ok=True)

    with open(prompt_file, "w") as f:
        f.write(prompt)

    # Determine model path (full or lite)
    ckpt_dir = os.path.join(hymotion_dir, "ckpts", "tencent")
    if os.path.exists(os.path.join(ckpt_dir, "HY-Motion-1.0")):
        model_path = os.path.join(ckpt_dir, "HY-Motion-1.0")
    elif os.path.exists(os.path.join(ckpt_dir, "HY-Motion-1.0-Lite")):
        model_path = os.path.join(ckpt_dir, "HY-Motion-1.0-Lite")
    else:
        raise FileNotFoundError("HY-Motion checkpoint not found in ckpts/tencent/")

    # Run inference via subprocess (cleanest way to handle HY-Motion's imports)
    result = subprocess.run(
        [
            sys.executable, "local_infer.py",
            "--model_path", model_path,
            "--input_text_dir", os.path.dirname(prompt_file),
            "--output_dir", output_dir,
            "--num_seeds", "1",
            "--disable_duration_est",
            "--disable_rewrite",
        ],
        cwd=hymotion_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        print(f"[HY-Motion] stderr: {result.stderr[:500]}")
        raise RuntimeError(f"HY-Motion inference failed: {result.stderr[:200]}")

    # Find output .npy file
    import glob
    npy_files = sorted(glob.glob(os.path.join(output_dir, "**/*.npy"), recursive=True))
    if not npy_files:
        raise FileNotFoundError("No .npy output from HY-Motion")

    joints = np.load(npy_files[-1])  # Shape: (num_frames, 22, 3) typically
    # Clean up
    os.remove(prompt_file)

    return joints.tolist()


def _run_momask_inference(prompt: str, num_frames: int = 60) -> list:
    """Run MoMask inference. Returns list of frames with joint positions."""
    import sys
    import torch

    momask_dir = motion_model["dir"]
    if momask_dir not in sys.path:
        sys.path.insert(0, momask_dir)

    # MoMask has a generate.py interface
    from models.mask_transformer.transformer import MaskTransformer
    device = motion_model.get("device", "cuda")

    # Use MoMask's generation pipeline
    result = subprocess.run(
        [
            sys.executable, "gen_t2m.py",
            "--text", prompt,
            "--length", str(num_frames),
        ],
        cwd=momask_dir,
        capture_output=True, text=True, timeout=60,
    )

    # Parse output
    import glob
    npy_files = sorted(glob.glob(os.path.join(momask_dir, "output/**/*.npy"), recursive=True))
    if npy_files:
        joints = np.load(npy_files[-1])
        return joints.tolist()
    raise RuntimeError("MoMask inference produced no output")


def _run_mld_inference(prompt: str, num_frames: int = 60) -> list:
    """Run MLD inference. Returns list of frames with joint positions."""
    import torch
    from mld.models.modeltype.mld import MLD as MLDModel
    device = motion_model.get("device", "cpu")
    model = MLDModel.load_from_checkpoint(motion_model["checkpoint"]).to(device)
    model.eval()
    with torch.no_grad():
        result = model.generate(prompt, num_frames=num_frames)
    return result.cpu().numpy().tolist()


def _joints_to_mp33(joints_3d: list, frame_width: int = 640,
                     frame_height: int = 480) -> list:
    """Convert SMPL 22 joints (3D) to MediaPipe 33-point format (2D + vis).

    Projects 3D joints to 2D using simple orthographic projection,
    then maps SMPL indices to MediaPipe indices.
    """
    frames = []
    for frame_joints in joints_3d:
        mp33 = [(frame_width / 2, frame_height / 2, 0.5)] * 33

        for smpl_idx, mp_idx in SMPL_TO_MP33.items():
            if smpl_idx < len(frame_joints):
                j = frame_joints[smpl_idx]
                # Orthographic projection: use x, y (ignore z for 2D)
                x = j[0] * frame_width * 0.3 + frame_width / 2
                y = -j[1] * frame_height * 0.3 + frame_height / 2  # flip Y
                mp33[mp_idx] = (round(x, 2), round(y, 2), 0.95)

        # Fill unmapped points via interpolation
        # Eyes, ears from head position
        if mp33[0][2] > 0.5:  # nose exists
            nx, ny = mp33[0][0], mp33[0][1]
            mp33[1] = (nx - 8, ny - 10, 0.9)   # left eye inner
            mp33[2] = (nx - 12, ny - 8, 0.9)    # left eye
            mp33[3] = (nx - 16, ny - 6, 0.9)    # left eye outer
            mp33[4] = (nx + 8, ny - 10, 0.9)    # right eye inner
            mp33[5] = (nx + 12, ny - 8, 0.9)    # right eye
            mp33[6] = (nx + 16, ny - 6, 0.9)    # right eye outer
            mp33[7] = (nx - 25, ny, 0.9)         # left ear
            mp33[8] = (nx + 25, ny, 0.9)         # right ear
            mp33[9] = (nx - 5, ny + 15, 0.9)     # mouth left
            mp33[10] = (nx + 5, ny + 15, 0.9)    # mouth right

        # Hands from wrists
        for wrist_idx, base_idx in [(15, 17), (16, 18)]:
            if mp33[wrist_idx][2] > 0.5:
                wx, wy = mp33[wrist_idx][0], mp33[wrist_idx][1]
                sign = -1 if wrist_idx == 15 else 1
                mp33[base_idx + 0] = (wx + sign * 5, wy + 8, 0.9)   # pinky
                mp33[base_idx + 1] = (wx + sign * 3, wy + 12, 0.9)  # index
                mp33[base_idx + 2] = (wx - sign * 5, wy + 5, 0.9)   # thumb
                mp33[base_idx + 3] = (wx + sign * 4, wy + 10, 0.9)  # pinky tip

        # Feet from ankles
        for ankle_idx, foot_idx, heel_idx in [(27, 31, 29), (28, 32, 30)]:
            if mp33[ankle_idx][2] > 0.5:
                ax, ay = mp33[ankle_idx][0], mp33[ankle_idx][1]
                sign = -1 if ankle_idx == 27 else 1
                mp33[foot_idx] = (ax + sign * 8, ay + 10, 0.9)
                mp33[heel_idx] = (ax - sign * 5, ay + 5, 0.9)

        frames.append([{"x": p[0], "y": p[1], "vis": p[2]} for p in mp33])

    return frames


from pydantic import BaseModel as PydanticModel

class MotionRequest(PydanticModel):
    prompt: str
    num_frames: int = 60
    fps: int = 30


@app.post("/generate_motion")
async def generate_motion(req: MotionRequest):
    """Generate a skeleton motion sequence from a text description.

    Uses Motion Diffusion Model (MDM) or Motion Latent Diffusion (MLD)
    to generate biomechanically accurate 3D motion from text.

    Returns:
        keypoints: list of frames, each frame is list of 33 keypoints
                   in MediaPipe format [{x, y, vis}, ...]
    """
    global motion_request_count

    if motion_model is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "No motion generation model loaded",
                "hint": "Run: bash dgx/setup_motion.sh to install MDM",
                "available_models": ["mdm", "mld", "t2m-gpt"],
            }
        )

    t0 = time.time()

    try:
        # Modal: proxy to remote A100 GPU
        if motion_model_type == "modal":
            return await _run_modal_inference(req.prompt, req.num_frames)

        if motion_model_type == "hymotion":
            joints_3d = _run_hymotion_inference(req.prompt, req.num_frames)
        elif motion_model_type == "momask":
            joints_3d = _run_momask_inference(req.prompt, req.num_frames)
        elif motion_model_type == "mld":
            joints_3d = _run_mld_inference(req.prompt, req.num_frames)
        else:
            return JSONResponse(status_code=503, content={"error": f"Unknown model type: {motion_model_type}"})

        # Convert 3D joints to MediaPipe 33-point 2D format
        keypoints = _joints_to_mp33(joints_3d)
        generation_ms = (time.time() - t0) * 1000
        motion_request_count += 1

        return {
            "prompt": req.prompt,
            "num_frames": len(keypoints),
            "fps": req.fps,
            "model": motion_model_type,
            "device": "NVIDIA DGX Spark",
            "generation_ms": round(generation_ms, 1),
            "keypoints": keypoints,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Motion generation failed: {str(e)}"}
        )


@app.get("/")
async def root():
    return {
        "service": "AEGIS DGX Inference — Pose Estimation + Motion Generation on NVIDIA DGX Spark",
        "models": {
            "pose": "YOLOv8n-pose (17 body keypoints)",
            "motion": motion_model_type or "not loaded (run setup_motion.sh)",
        },
        "hardware": f"NVIDIA DGX Spark — {platform.machine()} Grace ARM ({os.cpu_count()} cores)",
        "endpoints": {
            "/predict": "POST image → body keypoints per person",
            "/generate_motion": "POST {prompt, num_frames} → skeleton motion sequence",
            "/health": "GET server status",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 60)
    print("  AEGIS DGX Inference Server")
    print("  YOLOv8n-pose on NVIDIA DGX Spark")
    print(f"  CPU: {platform.machine()} ({os.cpu_count()} cores)")
    print(f"  Endpoint: http://{args.host}:{args.port}/predict")
    print("=" * 60)

    load_models()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
