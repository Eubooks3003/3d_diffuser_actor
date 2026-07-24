#!/bin/bash
# Create a TRAINING-ONLY conda env for the 3DDA mimicgen tokenization sweep.
# No robotics/sim (no robosuite/robomimic/mujoco/mimicgen) — the remote does not
# run in-loop rollouts (--rollout_freq 0). Order matters: torch is pinned first
# so flash-attn / other pip installs cannot clobber it.
set -e
ENV=${1:-3dda_train}

conda create -y -n "$ENV" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# 0) a fresh pip — old pip cannot read torch 2.x wheel tags (manylinux_2_28) and
#    fails with "No matching distribution found for torch==2.4.1", only seeing
#    versions up to 2.0.1. This is the usual cause of that error.
python -m pip install --upgrade pip
echo "python=$(python --version 2>&1)  pip=$(pip --version)"

# 1) torch FIRST. aarch64 (GH200) CUDA wheels live on the cu124 index and only
#    for newer versions, so don't hard-pin there; x86_64 matches the local stack.
if [ "$(uname -m)" = "aarch64" ]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
fi

# 2) flash-attn is OPTIONAL — the attention layers fall back to torch SDPA
#    (which dispatches to FlashAttention-2 kernels on Ampere/Hopper anyway).
#    On aarch64 / GH200 there is no prebuilt wheel and the source build is slow
#    and fragile, so we attempt it best-effort and continue on failure.
pip install flash-attn --no-build-isolation || \
  echo "flash-attn not installed — using torch SDPA fallback (fine on GH200)."

# 3) everything else (won't upgrade torch)
pip install -r requirements_train.txt

echo
echo "Env '$ENV' ready. Sanity check:"
echo "  conda activate $ENV && python -c 'from diffuser_actor import DiffuserActor; print(\"import OK\")'"
