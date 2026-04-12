#!/bin/bash
# =============================================================================
# slurm/eval_all.sh
#
# Run scripts/run_eval_all.py as a SLURM batch job.
# Evaluates and calibrates all completed checkpoints, then prints a summary.
#
# Submit:
#   sbatch slurm/eval_all.sh
#
# Submit after training array completes:
#   sbatch --dependency=afterok:<train_job_id> slurm/eval_all.sh
#
# Monitor:
#   tail -f /gscratch/scrubbed/sanmarco/equivariant-sar/logs/eval_all.log
# =============================================================================

#SBATCH --job-name=sar-eval
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/eval_all.log
#SBATCH --error=/gscratch/scrubbed/sanmarco/equivariant-sar/logs/eval_all.log

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIF=/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
VENV=/gscratch/scrubbed/sanmarco/venv
PROJECT=/gscratch/scrubbed/sanmarco/equivariant-sar

set -euo pipefail
mkdir -p "${PROJECT}/logs" "${PROJECT}/results"

echo "======================================================"
echo "eval_all: evaluate + calibrate all completed runs"
echo "Node: $(hostname)"
echo "======================================================"

apptainer exec \
    --nv \
    --bind /gscratch \
    "${SIF}" \
    /bin/bash -c "
set -euo pipefail
source ${VENV}/bin/activate
cd ${PROJECT}

python scripts/run_eval_all.py \
    --num-workers 4 \
    --batch-size  256
"

echo "eval_all finished."
