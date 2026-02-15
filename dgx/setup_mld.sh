#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Setup MLD (Motion Latent Diffusion) on DGX Spark
# Run ON THE DGX after SSH-ing in:
#   ssh asus@gx10-eb94
#   bash setup_mld.sh
# ═══════════════════════════════════════════════════════════════════════

set -e
echo "Setting up MLD on DGX Spark..."

# Use existing venv
cd ~/aegis-dgx
source bin/activate

# 1. Install MLD dependencies
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || echo "PyTorch already installed"
pip install -q transformers sentencepiece scipy tqdm omegaconf einops

# 2. Clone MLD repo
if [ ! -d "MLD" ]; then
    git clone https://github.com/ChenFengYe/motion-latent-diffusion.git MLD
fi

# 3. Download pre-trained model (~200MB)
mkdir -p models
cd MLD
if [ ! -f "../models/mld_humanml3d.ckpt" ]; then
    echo "Downloading MLD checkpoint..."
    # Try gdown first, then wget
    pip install -q gdown 2>/dev/null
    gdown "https://drive.google.com/uc?id=1U0FJXqjgeBS0cCP49o4hkLz-8Nkg2mY-" -O ../models/mld_humanml3d.ckpt 2>/dev/null || \
    echo "⚠ Auto-download failed. Download manually from:"
    echo "  https://drive.google.com/drive/folders/1U93FRqTbxMQ9UD_qBPrFg6DC5PseyYYI"
    echo "  Place at: ~/aegis-dgx/models/mld_humanml3d.ckpt"
fi

# 4. Download SMPL body model
bash prepare/download_smpl_model.sh 2>/dev/null || echo "SMPL download skipped"

# 5. Download T5 for text encoding
python -c "from transformers import T5Tokenizer, T5EncoderModel; T5Tokenizer.from_pretrained('t5-base'); T5EncoderModel.from_pretrained('t5-base'); print('T5 cached')"

cd ..

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  MLD setup complete!"
echo "  Now update inference_server.py and restart:"
echo "  python inference_server.py --port 8080"
echo "═══════════════════════════════════════════════════════"
