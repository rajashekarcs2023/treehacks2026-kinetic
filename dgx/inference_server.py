"""
AEGIS DGX Inference Server — Whole-Body Pose Estimation on NVIDIA DGX Spark.

Uses YOLOv8n-pose for 17-keypoint body pose + ultralytics framework,
running on NVIDIA DGX Spark's 72-core Grace ARM CPU.

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

app = FastAPI(title="AEGIS DGX Inference", version="1.0")

# ── Global model ─────────────────────────────────────────────
pose_model = None
request_count = 0
total_inference_ms = 0

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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": pose_model is not None,
        "model": "YOLOv8n-pose (17 body keypoints)",
        "platform": f"{platform.machine()} ({os.cpu_count()} cores)",
        "device": "NVIDIA DGX Spark (Grace ARM CPU)",
        "total_requests": request_count,
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


@app.get("/")
async def root():
    return {
        "service": "AEGIS DGX Inference — Pose Estimation on NVIDIA DGX Spark",
        "model": "YOLOv8n-pose (17 body keypoints)",
        "hardware": f"NVIDIA DGX Spark — {platform.machine()} Grace ARM ({os.cpu_count()} cores)",
        "endpoints": {
            "/predict": "POST image → body keypoints per person",
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
