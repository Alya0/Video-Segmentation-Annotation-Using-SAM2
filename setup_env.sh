#!/usr/bin/env bash
set -e  # Exit immediately if a command fails

ENV_NAME="segmentation_app"
PYTHON_VERSION="3.11"

echo "🚀 Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "🔄 Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "🎥 Installing ffmpeg"
conda install -c conda-forge ffmpeg -y

echo "🔥 Installing PyTorch with CUDA 12.4"
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
  -c pytorch -c nvidia -y

echo "📦 Installing Python packages via pip"
pip install --upgrade pip
pip install \
  imageio \
  gradio \
  opencv-python \
  python-dotenv \
  omegaconf \
  imageio-ffmpeg

echo "📥 Cloning SAM2 repository"
if [ ! -d "sam2" ]; then
  git clone https://github.com/facebookresearch/sam2.git
fi

echo "🔧 Installing SAM2 (editable mode)"
cd sam2
pip install -e .
cd ..

echo "📁 Moving into segmentation-app directory"
cd segmentation-app

echo "✅ Environment setup complete!"
