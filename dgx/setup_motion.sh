#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# AEGIS DGX Motion Generation Setup
# Sets up MDM (Motion Diffusion Model) on NVIDIA DGX Spark
#
# Usage: SSH into DGX, then run:
#   bash dgx/setup_motion.sh
#
# After setup, restart the inference server:
#   python dgx/inference_server.py --port 8080
# ═══════════════════════════════════════════════════════════════════════

set -e
echo "═══════════════════════════════════════════════════════"
echo "  AEGIS — Setting up Motion Diffusion Model on DGX"
echo "═══════════════════════════════════════════════════════"

# 1. Clone MDM
if [ ! -d "motion-diffusion-model" ]; then
    echo "[1/5] Cloning MDM repository..."
    git clone https://github.com/GuyTevet/motion-diffusion-model.git
else
    echo "[1/5] MDM directory exists, skipping clone"
fi

cd motion-diffusion-model

# 2. Install dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q numpy scipy matplotlib tqdm configargparse \
    clip@git+https://github.com/openai/CLIP.git \
    smplx chumpy trimesh mapbox_earcut

# MDM needs specific versions
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q transformers sentencepiece

# 3. Download HumanML3D text encoder + data
echo "[3/5] Setting up HumanML3D data..."
mkdir -p dataset
if [ ! -d "dataset/HumanML3D" ]; then
    # Download pre-processed text features (small, ~50MB)
    echo "  Downloading text features..."
    gdown "https://drive.google.com/uc?id=1PE41hSMBEUfnwGR4ZxPFJYMdoMcZ0YmB" -O dataset/humanml3d.zip 2>/dev/null || \
    wget -q "https://github.com/GuyTevet/motion-diffusion-model/releases/download/v0.1/humanml3d_text_features.tar.gz" -O dataset/humanml3d.tar.gz 2>/dev/null || \
    echo "  ⚠ Could not auto-download. Download manually from MDM repo."
fi

# 4. Download model weights
echo "[4/5] Downloading MDM model weights..."
mkdir -p ../models
if [ ! -f "../models/humanml_trans_enc_512.pt" ]; then
    # ~500MB model checkpoint
    wget -q "https://github.com/GuyTevet/motion-diffusion-model/releases/download/v0.1/humanml_trans_enc_512.zip" -O ../models/mdm_weights.zip 2>/dev/null && \
    unzip -q ../models/mdm_weights.zip -d ../models/ 2>/dev/null && \
    rm ../models/mdm_weights.zip 2>/dev/null || \
    echo "  ⚠ Could not auto-download weights. Download manually:"
    echo "    https://github.com/GuyTevet/motion-diffusion-model/releases"
    echo "    Place at: models/humanml_trans_enc_512.pt"
fi

cd ..

# 5. Verify
echo "[5/5] Verifying setup..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  Device: cpu (DGX Grace ARM)')
import os
mdm_exists = os.path.exists('motion-diffusion-model/model')
weights_exist = os.path.exists('models/humanml_trans_enc_512.pt')
print(f'  MDM code: {\"✓\" if mdm_exists else \"✗\"}')
print(f'  MDM weights: {\"✓\" if weights_exist else \"✗ (download manually)\"}')
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete! Restart the inference server:"
echo "  python dgx/inference_server.py --port 8080"
echo ""
echo "  Test motion generation:"
echo "  curl -X POST http://localhost:8080/generate_motion \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"a person doing a squat\", \"num_frames\": 60}'"
echo "═══════════════════════════════════════════════════════"
