#!/bin/bash
# Run ALL 7 tokenization experiments sequentially on one GPU (single-server case).
# For one-experiment-per-server, use scripts/train_one_experiment.sh instead.
# The per-experiment token configs live in train_one_experiment.sh (one source).
#
# Usage:
#   bash scripts/train_mimicgen_tokenization_sweep.sh [task] [gpu] [iters] [dataset]
set -e
HERE="$(dirname "$0")"
TASK=${1:-stack_d0}
GPU=${2:-0}
ITERS=${3:-100000}
DATASET=${4:-/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda}

for EXP in baseline single_action uniform random1 random2 no_proprio single_proprio; do
  bash "$HERE/train_one_experiment.sh" "$EXP" "$TASK" "$GPU" "$ITERS" "$DATASET"
done

echo "sweep complete — 7 runs. Check learning (no simulator needed):"
echo "  for r in baseline single_action uniform random1 random2 no_proprio single_proprio; do"
echo "    python scripts/check_learning.py --checkpoint train_logs/mimicgen/${TASK}_tok_\$r/last.pth \\"
echo "      --dataset $DATASET --task $TASK; done"
