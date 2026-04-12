#!/bin/bash
# =============================================================================
# slurm/setup_venv.sh
#
# One-time setup: create a venv on top of the NVIDIA PyTorch container that
# adds the packages not bundled in pytorch_24.12-py3.sif.
#
# Uses --system-site-packages so torch/torchvision/numpy come from the
# container — no re-download needed. lie-learn==0.0.2 (July 2024) is fully
# numpy 2.x compatible so no numpy pin is required.
#
# Run this interactively on a GPU node (NOT the login node):
#
#   salloc -A demo -p ckpt --gres=gpu:1 --mem=16G --time=0:30:00
#   bash /gscratch/scrubbed/sanmarco/equivariant-sar/slurm/setup_venv.sh
# =============================================================================

set -euo pipefail

SIF=/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
VENV=/gscratch/scrubbed/sanmarco/venv
PROJECT=/gscratch/scrubbed/sanmarco/equivariant-sar

INNER_SCRIPT=$(mktemp /tmp/setup_inner_XXXXXX.sh)
trap "rm -f ${INNER_SCRIPT}" EXIT

cat > "${INNER_SCRIPT}" << 'INNER'
#!/bin/bash
set -euo pipefail

VENV=/gscratch/scrubbed/sanmarco/venv

echo "=== Creating venv (--system-site-packages) at ${VENV} ==="
python -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"

echo "Python : $(which python)"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "NumPy  : $(python -c 'import numpy as np; print(np.__version__)')"

pip install --upgrade pip --quiet

# 1. lie-learn 0.0.2 — numpy 2.x compatible (fix landed July 2024)
echo "--- [1/3] lie-learn==0.0.2 + escnn dependencies ---"
pip install autograd pymanopt py3nj --quiet
pip install lie-learn==0.0.2 --no-build-isolation --quiet

# 2. escnn
echo "--- [2/3] escnn ---"
pip install escnn --quiet
pip install e2cnn --quiet

# 3. ML / data stack
echo "--- [3/3] ml + data packages ---"
pip install \
    "numpy<2" \
    wandb \
    scikit-learn \
    matplotlib \
    seaborn \
    rasterio \
    geopandas \
    sentinelsat \
    pandas \
    tifffile \
    --quiet

# Verify
echo ""
echo "=== Verifying installs ==="
python -c "
import numpy as np
import torch
import escnn
import sklearn, matplotlib, wandb, rasterio, pandas, tifffile

print('numpy     :', np.__version__)
assert tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 0), 'FAIL: numpy>=2 still present'
print('torch     :', torch.__version__)
print('cuda      :', torch.cuda.is_available())
print('escnn     : OK')
print('sklearn   : OK')
print('matplotlib: OK')
print('wandb     : OK')
print('rasterio  : OK')
print('pandas    : OK')
print('tifffile  : OK')
print('')
print('=== Setup complete ===')
"
INNER

echo "=== Running install inside container ==="
apptainer exec \
    --nv \
    --bind /gscratch \
    --bind "${INNER_SCRIPT}:${INNER_SCRIPT}" \
    "${SIF}" \
    /bin/bash "${INNER_SCRIPT}"

echo ""
echo "Next step — verify equivariance tests pass:"
echo "  apptainer exec --nv --bind /gscratch ${SIF} \\"
echo "    /bin/bash -c 'source ${VENV}/bin/activate && cd ${PROJECT} && python -m tests.test_equivariance'"
