#!/bin/bash
# Run ONE tokenization experiment (one token config) — for one-experiment-per-server.
# Option B (--diffuse_gripper): action_dim=10 = [pos(3), rot6d(6), gripper(1)].
# No in-loop rollouts (--rollout_freq 0); confirm learning with check_learning.py.
#
# Usage:
#   bash scripts/train_one_experiment.sh <experiment> [task] [gpu] [iters] [dataset]
#   experiment in:
#     baseline single_action uniform random1 random2 no_proprio single_proprio
#
# Example (server A): bash scripts/train_one_experiment.sh baseline
#          (server B): bash scripts/train_one_experiment.sh single_action
set -e
EXP=${1:?experiment required: baseline|single_action|uniform|random1|random2|no_proprio|single_proprio}
# TASKS: a single task, a quoted space-separated list, or "all12" (multitask).
TASKS=${2:-stack_d0}
GPU=${3:-0}
ITERS=${4:-100000}
DATASET=${5:-/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda}
PORT=$((29600 + GPU))

# Throughput knobs (env overrides) — defaults tuned for a GH200 (batch 16 badly
# underuses it). Bigger batch => fewer steps for the same epochs, so the ITERS
# default is 100k (NOT 300k): 300k steps at batch 64 trains ~4x too long.
# LR is sqrt-scaled from 1e-4@16 -> 2e-4@64. Keep these IDENTICAL across all 7
# runs so the tokenization comparison stays apples-to-apples.
BATCH=${BATCH:-64}
VAL_BATCH=${VAL_BATCH:-32}
WORKERS=${WORKERS:-16}
LR=${LR:-2e-4}

ALL12="coffee_d0 coffee_preparation_d0 hammer_cleanup_d0 kitchen_d0 mug_cleanup_d0 nut_assembly_d0 pick_place_d0 square_d0 stack_d0 stack_three_d0 threading_d0 three_piece_assembly_d0"
[ "$TASKS" = "all12" ] && TASKS="$ALL12"
# run tag: the task name for single-task, "multitask" for >1 (task_id embedding
# is enabled automatically; token groups are identical either way).
if [ "$(echo $TASKS | wc -w)" -gt 1 ]; then TAG="multitask"; else TAG="$TASKS"; fi

# groups sum to 10 = pos(3)+rot6d(6)+gripper(1). Random *proprio* groups match
# EC-Diffuser's randgroupA/B (10-dim); random *action* groups are 10-dim analogues.
case "$EXP" in
  baseline)        ACT="[3,6,1]";   PROP="[3,6,1]";  EXTRA="" ;;
  single_action)   ACT="[10]";      PROP="[3,6,1]";  EXTRA="" ;;
  uniform)         ACT="per_dim";   PROP="per_dim";  EXTRA="" ;;
  random1)         ACT="[2,1,4,3]"; PROP="[4,1,2,3]"; EXTRA="" ;;
  random2)         ACT="[3,1,2,4]"; PROP="[2,3,1,4]"; EXTRA="" ;;
  no_proprio)      ACT="[3,6,1]";   PROP="default";  EXTRA="--no_proprio 1" ;;
  single_proprio)  ACT="[3,6,1]";   PROP="[10]";     EXTRA="" ;;
  *) echo "unknown experiment: $EXP"; exit 1 ;;
esac

# timestamped subfolder: relaunching the same experiment never overwrites a
# previous run — each lands in train_logs/mimicgen/<run>/<stamp>/
STAMP=$(date +%Y%m%d_%H%M%S)
RUN="${TAG}_tok_${EXP}"
mkdir -p train_logs
echo "=== $RUN/$STAMP  (action=$ACT proprio=$PROP $EXTRA)  tasks=[$TASKS]  gpu=$GPU iters=$ITERS batch=$BATCH lr=$LR workers=$WORKERS ==="
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node 1 --master_port $PORT \
  main_trajectory_mimicgen.py \
  --tasks $TASKS --dataset "$DATASET" \
  --batch_size "$BATCH" --batch_size_val "$VAL_BATCH" --lr "$LR" \
  --train_iters "$ITERS" --val_freq 2500 --val_iters 25 --num_workers "$WORKERS" \
  --diffuse_gripper 1 \
  --goal_actions 1 \
  --action_token_groups "$ACT" --proprio_token_groups "$PROP" \
  $EXTRA \
  --rollout_freq 0 \
  --run_log_dir "$RUN/$STAMP" \
  2>&1 | tee "train_logs/${RUN}_${STAMP}.log"
