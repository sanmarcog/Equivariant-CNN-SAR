#!/bin/bash
# =============================================================================
# slurm/setup_hyak.sh
#
# One-time environment setup on Hyak (klone).
# Run this interactively on a login node before submitting any jobs.
#
# Usage:
#   bash slurm/setup_hyak.sh
# =============================================================================
set -euo pipefail

echo "=== Setting up sar-equivariant conda environment on Hyak ==="

# --- Load required modules ---
module load cuda/11.8.0     # adjust to available CUDA version
module load gcc             # needed for some package builds

# --- Create conda environment ---
conda create -n sar-equivariant python=3.11 -y
conda activate sar-equivariant

# --- PyTorch (match CUDA version above) ---
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# --- escnn and its dependencies ---
conda install -c conda-forge gfortran -y
pip install py3nj
pip install lie-learn --no-build-isolation
pip install escnn --no-deps
pip install e2cnn    # escnn dependency

# --- Data / ML stack ---
pip install \
    numpy \
    scipy \
    scikit-learn \
    rasterio \
    pandas \
    matplotlib \
    wandb

echo ""
echo "=== Setup complete. Test with: ==="
echo "  conda activate sar-equivariant"
echo "  python -m tests.test_equivariance"
