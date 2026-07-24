#!/bin/bash
# Training-only conda env for the 3DDA mimicgen tokenization sweep.
# No robotics/sim (no robosuite/robomimic/mujoco/mimicgen); no in-loop rollouts.
# Idempotent + self-verifying — safe to run once per server.
#
# Usage:   bash setup_remote_train_env.sh [env_name]
#          INSTALL_FLASH=1 bash setup_remote_train_env.sh   # also try flash-attn
set -e
ENV=${1:-3dda_train}
source "$(conda info --base)/etc/profile.d/conda.sh"

# Idempotent: recreate cleanly if the env already exists.
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo "env '$ENV' exists — recreating"
  conda env remove -y -n "$ENV"
fi
conda create -y -n "$ENV" python=3.10
conda activate "$ENV"

python -m pip install --upgrade pip   # old pip can't see torch 2.x wheels
echo "arch=$(uname -m)  python=$(python --version 2>&1)  pip=$(pip --version | awk '{print $2}')"

# torch first, arch-aware (GH200/aarch64 CUDA wheels live on the cu124 index).
if [ "$(uname -m)" = "aarch64" ]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r requirements_train.txt

# flash-attn is OPTIONAL: the attention layers fall back to torch SDPA, which
# runs FlashAttention-2 kernels on Hopper anyway. Skipped by default because the
# aarch64 source build is slow/fragile. Set INSTALL_FLASH=1 to attempt it.
if [ "${INSTALL_FLASH:-0}" = "1" ]; then
  pip install flash-attn --no-build-isolation || \
    echo "flash-attn build failed — SDPA fallback is used (fine on GH200)."
fi

echo "=== verifying import (no flash-attn needed) ==="
python -c "from diffuser_actor import DiffuserActor; print('import OK')"
echo "=== env '$ENV' ready — activate with: conda activate $ENV ==="
