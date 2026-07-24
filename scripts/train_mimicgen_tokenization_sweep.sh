#!/bin/bash
# Tokenization sweep on mimicgen, porting EC-Diffuser's action/proprio token
# grouping experiments to 3D Diffuser Actor.
#
# 3DDA's diffused action is 9-dim [pos(3), rot6d(6)] -- openness is predicted
# from position features rather than diffused -- so groups partition 9 dims,
# not EC-Diffuser's 7.
#
# Usage:
#   bash scripts/train_mimicgen_tokenization_sweep.sh stack_d0 0
#     (arg1 = task, arg2 = CUDA device)

set -e
TASK=${1:-stack_d0}
GPU=${2:-0}
ITERS=${3:-100000}
PORT=$((29600 + GPU))

# name | action groups | proprio groups
EXPERIMENTS=(
  "baseline_default|default|default"
  "uniform_per_dim|per_dim|per_dim"
  "singleaction_tokenprop|[9]|[3,6]"
  "tokenaction_singleprop|[3,6]|[9]"
  "noproprio_uniform|per_dim|default"
  "randgroupA|[2,1,3,1,2]|[4,1,2,2]"
  "randgroupB|[3,1,1,2,2]|[2,3,1,3]"
)

for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME ACT PROP <<< "$exp"
  RUN="${TASK}_${NAME}"
  echo "=== $RUN  (action=$ACT proprio=$PROP) ==="
  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node 1 --master_port $PORT \
    main_trajectory_mimicgen.py \
    --tasks "$TASK" \
    --batch_size 16 --batch_size_val 8 \
    --train_iters "$ITERS" --val_freq 2500 --val_iters 25 \
    --num_workers 6 \
    --action_token_groups "$ACT" \
    --proprio_token_groups "$PROP" \
    --run_log_dir "$RUN" \
    2>&1 | tee "train_logs/${RUN}.log"
done

echo "sweep complete. Evaluate each with:"
echo "  python online_evaluation_mimicgen/eval_mimicgen.py \\"
echo "    --checkpoint train_logs/mimicgen/<run>/best.pth --task $TASK \\"
echo "    --action-token-groups <groups> --num-episodes 25"
