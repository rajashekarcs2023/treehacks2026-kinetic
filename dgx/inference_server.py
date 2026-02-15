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
        from ultralytics import YOLO
        # YOLOv8n-pose: 17 keypoints, fast, well-tested
        pose_model = YOLO("yolov8n-pose.pt")
        # Force CPU since GB10 GPU not yet supported by PyTorch
        pose_model.to("cpu")
        print("[DGX] YOLOv8n-pose loaded (17 body keypoints, CPU mode)")
        print(f"[DGX] Platform: {platform.machine()} ({os.cpu_count()} cores)")

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

    # Try MLD (Motion Latent Diffusion) — lighter, faster
    try:
        from mld.models.modeltype.mld import MLD
        from mld.config import parse_args as mld_parse_args
        import torch

        # Check for model weights
        mld_ckpt = os.environ.get("MLD_CHECKPOINT", "models/mld_humanml3d.ckpt")
        if os.path.exists(mld_ckpt):
            print(f"[DGX Motion] Loading MLD from {mld_ckpt}...")
            motion_model = {"type": "mld", "checkpoint": mld_ckpt}
            motion_model_type = "mld"
            print("[DGX Motion] MLD model ready")
            return
    except ImportError:
        pass

    # Try MDM (Motion Diffusion Model)
    try:
        mdm_dir = os.environ.get("MDM_DIR", "motion-diffusion-model")
        mdm_ckpt = os.environ.get("MDM_CHECKPOINT", "models/humanml_trans_enc_512.pt")
        if os.path.exists(os.path.join(mdm_dir, "model")) and os.path.exists(mdm_ckpt):
            print(f"[DGX Motion] MDM directory found at {mdm_dir}")
            motion_model = {"type": "mdm", "dir": mdm_dir, "checkpoint": mdm_ckpt}
            motion_model_type = "mdm"
            print("[DGX Motion] MDM model ready")
            return
    except Exception:
        pass

    # Try T2M-GPT
    try:
        t2m_ckpt = os.environ.get("T2M_CHECKPOINT", "models/t2m_gpt.pt")
        if os.path.exists(t2m_ckpt):
            motion_model = {"type": "t2m", "checkpoint": t2m_ckpt}
            motion_model_type = "t2m"
            print("[DGX Motion] T2M-GPT model ready")
            return
    except Exception:
        pass

    print("[DGX Motion] No motion generation model found — /generate_motion will return 503")
    print("[DGX Motion] Run: bash dgx/setup_motion.sh to install MDM")


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


def _run_mdm_inference(prompt: str, num_frames: int = 60) -> list:
    """Run MDM inference. Returns list of frames, each frame is list of 22 (x,y,z) joints."""
    import sys
    import torch

    mdm_dir = motion_model["dir"]
    if mdm_dir not in sys.path:
        sys.path.insert(0, mdm_dir)

    from utils.parser_util import generate_args
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    # Generate motion
    args = generate_args()
    args.model_path = motion_model["checkpoint"]
    args.num_samples = 1
    args.num_repetitions = 1
    args.motion_length = num_frames / 20.0  # MDM uses 20fps

    model, diffusion = create_model_and_diffusion(args, None)
    load_model_wo_clip(model, args.model_path)
    model.eval()

    # Encode text prompt
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

    sample = diffusion.p_sample_loop(
        model,
        (1, model.njoints, model.nfeats, num_frames),
        clip_denoised=False,
        model_kwargs={"y": {"text": [prompt], "lengths": torch.tensor([num_frames])}},
        skip_timesteps=0,
        progress=True,
    )

    # Convert to joint positions
    joints = recover_from_ric(sample.squeeze().permute(1, 0), 22)
    return joints.cpu().numpy().tolist()


def _run_mld_inference(prompt: str, num_frames: int = 60) -> list:
    """Run MLD inference. Returns list of frames with joint positions."""
    import torch

    # MLD uses a simpler interface
    from mld.models.modeltype.mld import MLD as MLDModel
    model = MLDModel.load_from_checkpoint(motion_model["checkpoint"])
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
        if motion_model_type == "mdm":
            joints_3d = _run_mdm_inference(req.prompt, req.num_frames)
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
