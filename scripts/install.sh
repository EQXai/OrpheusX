#!/bin/bash
set -e
# install dependencies for Orpheus TTS training inside a virtual environment

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# PyTorch version compatible with CUDA 12.4
TORCH_VER=2.6.0

VENV_DIR="$ROOT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer transformers==4.51.3 unsloth snac torchaudio ipython librosa soundfile whisperx

if command -v apt-get >/dev/null; then
  apt-get update && apt-get install -y ffmpeg
fi

# remove pytorch installed by unsloth
pip uninstall -y torch || true

echo "WARNING: This program is not compatible with CUDA versions higher than 12.4"
pip install torch==${TORCH_VER} torchvision==0.21.0 torchaudio==${TORCH_VER} --index-url https://download.pytorch.org/whl/cu124

echo "All dependencies installed with torch==${TORCH_VER}"
echo "Activate the environment with: source $VENV_DIR/bin/activate"

# Additional CUDA dependencies
if command -v apt >/dev/null; then
  sudo apt install libcudnn8 libcudnn8-dev
fi
