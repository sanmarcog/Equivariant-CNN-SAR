#!/bin/bash
# =============================================================================
# slurm/eval_array.sh
#
# Run evaluate.py + calibrate.py for all 24 completed training runs, then
# rsync results back to your Mac.
#
# Submit after all training jobs finish:
#   sbatch --dependency=afterok:<train_job_id> slurm/eval_array.sh
#
# Or run immediately if training is already done:
#   sbatch slurm/eval_array.sh
#
# Results:
#   After jobs finish, pull results to your Mac manually:
#   rsync -avz sanmarco@klone.hyak.uw.edu:/gscratch/scrubbed/sanmarco/equivariant-sar/results/ \
#     /Users/sanmarco/Documents/GitHub/Equivariant-CNN-SAR/results/
# =============================================================================

#SBATCH --job-name=sar-eval
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/eval_%a.log
#SBATCH --error=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/eval_%a.log

# ---------------------------------------------------------------------------
# Paths — edit MAC_IP before submitting
# ---------------------------------------------------------------------------
SIF=/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
VENV=/gscratch/scrubbed/sanmarco/venv
PROJECT=/gscratch/scrubbed/sanmarco/equivariant-sar

set -euo pipefail
mkdir -p "${PROJECT}/logs" "${PROJECT}/results"

# ---------------------------------------------------------------------------
# Map task ID → model and data fraction
# ---------------------------------------------------------------------------
MODELS=(c8 so2 d4 cnn aug resnet)
FRACTIONS=(0.1 0.25 0.5 1.0)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

MODEL="${MODELS[$MODEL_IDX]}"
FRACTION="${FRACTIONS[$FRAC_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: evaluate + calibrate  model=${MODEL}  fraction=${FRACTION}"

# ---------------------------------------------------------------------------
# W&B — read key from file so it's available inside the container
# ---------------------------------------------------------------------------
export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || echo "")
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "WARNING: ~/.wandb_api_key not found — W&B logging will be disabled"
fi

# ---------------------------------------------------------------------------
# Evaluate and calibrate inside container
# ---------------------------------------------------------------------------
apptainer exec \
    --nv \
    --bind /gscratch \
    "${SIF}" \
    /bin/bash -c "
set -euo pipefail
source ${VENV}/bin/activate
cd ${PROJECT}

python evaluate.py \
    --model          ${MODEL} \
    --data-fraction  ${FRACTION} \
    --val-csv        ${PROJECT}/data/splits/val.csv \
    --test-csv       ${PROJECT}/data/splits/test_ood.csv \
    --stats-path     ${PROJECT}/data/splits/norm_stats.json \
    --checkpoint-dir ${PROJECT}/checkpoints \
    --results-dir    ${PROJECT}/results \
    --batch-size     256 \
    --num-workers    4

python calibrate.py \
    --model          ${MODEL} \
    --data-fraction  ${FRACTION} \
    --results-dir    ${PROJECT}/results
"

echo "Task ${SLURM_ARRAY_TASK_ID} finished."

# ---------------------------------------------------------------------------
# To pull results to your Mac after all jobs finish, run this from your Mac:
#
#   rsync -avz sanmarco@klone.hyak.uw.edu:/gscratch/scrubbed/sanmarco/equivariant-sar/results/ \
#     /Users/sanmarco/Documents/GitHub/Equivariant-CNN-SAR/results/
#
# Hyak cannot reach your Mac directly (different networks), so push from Mac.
# ---------------------------------------------------------------------------
