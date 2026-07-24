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

# 1) torch FIRST, pinned (cu121 wheels — matches the local training stack)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 2) flash-attn (hard dependency of the attention layers; needs torch present).
#    Ampere+ GPU required (A100/H100 on lambda are fine).
pip install flash-attn --no-build-isolation

# 3) everything else (won't upgrade torch)
pip install -r requirements_train.txt

echo
echo "Env '$ENV' ready. Sanity check:"
echo "  conda activate $ENV && python -c 'from diffuser_actor import DiffuserActor; print(\"import OK\")'"
