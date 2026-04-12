#!/bin/bash
# =============================================================================
# slurm/smoke_test.sh
#
# Single-job end-to-end smoke test: C8 equivariant model, 10% data, 2 epochs.
# Run this before submitting the full 24-job array to confirm the container,
# venv, data paths, and W&B all work correctly on Hyak.
#
# Submit:
#   sbatch slurm/smoke_test.sh
#
# Monitor:
#   tail -f /gscratch/scrubbed/sanmarco/equivariant-sar/logs/smoke_test.log
#
# Expected outcome:
#   - Two training epochs complete without error
#   - Checkpoint saved to checkpoints/c8_frac0p1/
#   - W&B run appears at wandb.ai (or WARNING printed if key missing)
#   - Final line: "Smoke test finished."
# =============================================================================

#SBATCH --job-name=sar-smoke
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/smoke_test.log
#SBATCH --error=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/smoke_test.log

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIF=/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
VENV=/gscratch/scrubbed/sanmarco/venv
PROJECT=/gscratch/scrubbed/sanmarco/equivariant-sar

set -euo pipefail
mkdir -p "${PROJECT}/logs" "${PROJECT}/checkpoints"

echo "======================================================"
echo "Smoke test: C8 model, 10% data, 2 epochs"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Container: ${SIF}"
echo "======================================================"

# ---------------------------------------------------------------------------
# CUDA capability check
# ---------------------------------------------------------------------------
MIN_CC=70
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ "$GPU_CC" -lt "$MIN_CC" ]; then
    echo "GPU compute capability ${GPU_CC} < ${MIN_CC} (sm_70 required). Requeueing job ${SLURM_JOB_ID}..."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
echo "GPU compute capability: ${GPU_CC} — OK"

# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || echo "")
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "WARNING: ~/.wandb_api_key not found — W&B logging will be disabled"
fi

# ---------------------------------------------------------------------------
# Run equivariance tests first — abort if they fail
# ---------------------------------------------------------------------------
echo ""
echo "--- Running equivariance tests ---"
apptainer exec \
    --nv \
    --bind /gscratch \
    "${SIF}" \
    /bin/bash -c "
source ${VENV}/bin/activate
cd ${PROJECT}
python -m tests.test_equivariance
"

echo ""
echo "--- Equivariance tests passed. Starting training smoke test ---"

# ---------------------------------------------------------------------------
# Train C8 for 2 epochs at 10% data
# ---------------------------------------------------------------------------
apptainer exec \
    --nv \
    --bind /gscratch \
    "${SIF}" \
    /bin/bash -c "
set -euo pipefail
source ${VENV}/bin/activate
cd ${PROJECT}

python train.py \
    --model          c8 \
    --data-fraction  0.1 \
    --epochs         2 \
    --batch-size     64 \
    --lr             1e-3 \
    --weight-decay   1e-4 \
    --pos-weight     3.0 \
    --patience       10 \
    --train-csv      ${PROJECT}/data/splits/train.csv \
    --val-csv        ${PROJECT}/data/splits/val.csv \
    --stats-path     ${PROJECT}/data/splits/norm_stats.json \
    --checkpoint-dir ${PROJECT}/checkpoints \
    --num-workers    4 \
    --wandb-project  equivariant-sar
"

echo ""
echo "Smoke test finished."
