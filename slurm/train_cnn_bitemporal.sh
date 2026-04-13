#!/bin/bash
# =============================================================================
# slurm/train_cnn_bitemporal.sh
#
# 4-job training array: cnn_bitemporal × 4 data fractions.
# Runs inside the NVIDIA PyTorch Apptainer container with a venv overlay.
#
# Job layout (SLURM_ARRAY_TASK_ID 0–3):
#   task_id = fraction_idx
#   fraction_idx : 0=0.1  1=0.25  2=0.5  3=1.0
#
# Bi-temporal notes:
#   - Uses the same 7-channel norm stats as d4_bitemporal.
#   - Norm stats path: ${PROJECT}/data/splits/norm_stats_bitemporal.json
#   - Dataset returns (post_5ch, pre_5ch) pairs; matched to CNNBiTemporal.
#
# Submit all 4 jobs:
#   sbatch slurm/train_cnn_bitemporal.sh
#
# Submit a single job (e.g. 100% data, task 3):
#   sbatch --array=3 slurm/train_cnn_bitemporal.sh
# =============================================================================

#SBATCH --job-name=sar-cnn-bt
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/cnn_bitemporal_%a.log
#SBATCH --error=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/cnn_bitemporal_%a.log
#SBATCH --requeue

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIF=/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
VENV=/gscratch/scrubbed/sanmarco/venv
PROJECT=/gscratch/scrubbed/sanmarco/equivariant-sar

set -euo pipefail
mkdir -p "${PROJECT}/logs" "${PROJECT}/checkpoints"

# ---------------------------------------------------------------------------
# Map task ID → data fraction
# ---------------------------------------------------------------------------
FRACTIONS=(0.1 0.25 0.5 1.0)
FRACTION="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: model=cnn_bitemporal  fraction=${FRACTION}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Container: ${SIF}"
echo "======================================================"

# ---------------------------------------------------------------------------
# CUDA capability check — PyTorch 2.6 requires sm_70+.
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
# Run inside container
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
    --model          cnn_bitemporal \
    --data-fraction  ${FRACTION} \
    --epochs         100 \
    --batch-size     64 \
    --lr             1e-3 \
    --weight-decay   1e-4 \
    --pos-weight     3.0 \
    --patience       10 \
    --train-csv      ${PROJECT}/data/splits/train.csv \
    --val-csv        ${PROJECT}/data/splits/val.csv \
    --stats-path     ${PROJECT}/data/splits/norm_stats.json \
    --bitemporal-stats-path ${PROJECT}/data/splits/norm_stats_bitemporal.json \
    --checkpoint-dir ${PROJECT}/checkpoints \
    --num-workers    8 \
    --wandb-project  equivariant-sar
"

echo "Task ${SLURM_ARRAY_TASK_ID} (cnn_bitemporal frac=${FRACTION}) finished."
