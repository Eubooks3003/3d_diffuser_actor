#!/bin/bash
# Tokenization sweep on mimicgen — EC-Diffuser action/proprio grouping ported to
# 3D Diffuser Actor, Option B (--diffuse_gripper): openness is folded into the
# diffused action (action_dim 9 -> 10) so a dedicated gripper token exists.
#
# All 10 dims = [pos(3), rot6d(6), gripper(1)]; groups partition 10.
# NO in-loop rollouts (--rollout_freq 0): the remote does not run a simulator.
# Confirm learning with scripts/check_learning.py instead.
#
# Usage:
#   bash scripts/train_mimicgen_tokenization_sweep.sh <task> <gpu> <iters> <dataset>
#     e.g. bash scripts/train_mimicgen_tokenization_sweep.sh stack_d0 0 100000 \
#            /lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda
set -e
TASK=${1:-stack_d0}
GPU=${2:-0}
ITERS=${3:-100000}
DATASET=${4:-/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda}
PORT=$((29600 + GPU))

# name | action groups | proprio groups | extra flags
#   groups sum to 10 = pos(3)+rot6d(6)+gripper(1). Random groups keep 4 arbitrary
#   partitions; the proprio random groups match EC-Diffuser's randgroupA/B (both
#   10-dim), the action random groups are 10-dim analogues (EC's action was 7-dim).
EXPERIMENTS=(
  "baseline|[3,6,1]|[3,6,1]|"
  "single_action|[10]|[3,6,1]|"
  "uniform|per_dim|per_dim|"
  "random1|[2,1,4,3]|[4,1,2,3]|"
  "random2|[3,1,2,4]|[2,3,1,4]|"
  "no_proprio|[3,6,1]|default|--no_proprio 1"
  "single_proprio|[3,6,1]|[10]|"
)

for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME ACT PROP EXTRA <<< "$exp"
  RUN="${TASK}_tok_${NAME}"
  echo "=== $RUN  (action=$ACT proprio=$PROP $EXTRA) ==="
  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node 1 --master_port $PORT \
    main_trajectory_mimicgen.py \
    --tasks "$TASK" \
    --dataset "$DATASET" \
    --batch_size 16 --batch_size_val 8 \
    --train_iters "$ITERS" --val_freq 2500 --val_iters 25 \
    --num_workers 6 \
    --diffuse_gripper 1 \
    --action_token_groups "$ACT" \
    --proprio_token_groups "$PROP" \
    $EXTRA \
    --rollout_freq 0 \
    --run_log_dir "$RUN" \
    2>&1 | tee "train_logs/${RUN}.log"
done

echo "sweep complete — 7 runs. Check learning (no simulator needed):"
echo "  for r in baseline single_action uniform random1 random2 no_proprio single_proprio; do"
echo "    python scripts/check_learning.py --checkpoint train_logs/mimicgen/${TASK}_tok_\$r/last.pth \\"
echo "      --dataset $DATASET --task $TASK; done"
