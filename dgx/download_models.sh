#!/bin/bash
# Download RTMPose-WholeBody ONNX model for DGX Spark inference
set -e

MODEL_DIR="$(dirname "$0")/models"
mkdir -p "$MODEL_DIR"

echo "=== Downloading RTMPose-WholeBody ONNX model ==="

# RTMPose-X WholeBody (body + hands + face = 133 keypoints)
# Source: OpenMMLab / MMPose
POSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
POSE_FILE="$MODEL_DIR/rtmpose_wholebody.onnx"

if [ ! -f "$POSE_FILE" ]; then
    echo "Downloading RTMPose-WholeBody..."
    # Try direct ONNX first (if available)
    wget -q "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-wholebody_pt-body7_700e-384x288-401dfc90_20230629.onnx" \
        -O "$POSE_FILE" 2>/dev/null && echo "Downloaded ONNX directly" || {
        echo "Direct ONNX not found, trying zip..."
        wget -q "$POSE_URL" -O "$MODEL_DIR/rtmpose.zip" 2>/dev/null && {
            cd "$MODEL_DIR"
            unzip -o rtmpose.zip
            # Find the .onnx file in extracted contents
            find . -name "*.onnx" -exec mv {} rtmpose_wholebody.onnx \;
            rm -f rtmpose.zip
            echo "Extracted ONNX from zip"
        } || {
            echo ""
            echo "=== AUTO-DOWNLOAD FAILED ==="
            echo "Please manually download an RTMPose WholeBody ONNX model."
            echo "Options:"
            echo "  1. pip install mim && mim download mmpose --config rtmpose-x_8xb32-270e_wholebody-384x288"
            echo "  2. Export from PyTorch: python -m mmpose.tools.deployment.pytorch2onnx ..."
            echo "  3. HuggingFace: search 'rtmpose wholebody onnx'"
            echo ""
            echo "Place the .onnx file at: $POSE_FILE"
        }
    }
else
    echo "RTMPose model already exists at $POSE_FILE"
fi

# Optional: YOLOv8n for person detection (improves accuracy)
DET_FILE="$MODEL_DIR/yolov8n.onnx"
if [ ! -f "$DET_FILE" ]; then
    echo "Downloading YOLOv8n (person detection)..."
    pip install -q ultralytics 2>/dev/null
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640)
import shutil
shutil.move('yolov8n.onnx', '$DET_FILE')
print('YOLOv8n ONNX exported')
" 2>/dev/null || {
        echo "YOLOv8n download failed — will use full-image pose (still works)"
    }
else
    echo "YOLOv8n already exists at $DET_FILE"
fi

echo ""
echo "=== Model setup complete ==="
ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || echo "No .onnx files found — check errors above"
