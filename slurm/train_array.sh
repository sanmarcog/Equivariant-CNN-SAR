#!/bin/bash
# =============================================================================
# slurm/train_array.sh
#
# 24-job training array: 6 models × 4 data fractions.
# Runs inside the NVIDIA PyTorch Apptainer container with a venv overlay for
# packages not bundled in the container (escnn, wandb, rasterio, etc.).
#
# Job layout (SLURM_ARRAY_TASK_ID 0–23):
#   task_id = model_idx * 4 + fraction_idx
#   model_idx    : 0=c8  1=so2  2=d4  3=cnn  4=aug  5=resnet
#   fraction_idx : 0=0.1  1=0.25  2=0.5  3=1.0
#
# Preemption safety:
#   train.py saves a checkpoint after every epoch.
#   If a job is preempted and re-queued, it resumes from the last checkpoint
#   automatically — no extra flags needed.
#
# Prerequisites:
#   1. Run slurm/setup_venv.sh once to create the venv.
#   2. Confirm equivariance tests pass inside the container.
#
# Submit all 24 jobs:
#   sbatch slurm/train_array.sh
#
# Submit a single job (e.g. c8 at 100% data, task 3):
#   sbatch --array=3 slurm/train_array.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /gscratch/scrubbed/sanmarco/equivariant-sar/logs/train_3.log
# =============================================================================

#SBATCH --job-name=sar-train
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/train_%a.log
#SBATCH --error=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/train_%a.log
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
# Map task ID → model and data fraction
# ---------------------------------------------------------------------------
MODELS=(c8 so2 d4 cnn aug resnet)
FRACTIONS=(0.1 0.25 0.5 1.0)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

MODEL="${MODELS[$MODEL_IDX]}"
FRACTION="${FRACTIONS[$FRAC_IDX]}"

echo "======================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL}  fraction=${FRACTION}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Container: ${SIF}"
echo "======================================================"

# ---------------------------------------------------------------------------
# CUDA capability check — PyTorch 2.6 requires sm_70+ (Volta and newer).
# P100 (sm_60) and older cards will silently fail or produce wrong results.
# Requeue the job so SLURM retries on a compatible node.
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
# W&B — read key from file so it's available inside the container
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
    --model          ${MODEL} \
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
    --checkpoint-dir ${PROJECT}/checkpoints \
    --num-workers    8 \
    --wandb-project  equivariant-sar
"

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
