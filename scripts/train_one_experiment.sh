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
TASK=${2:-stack_d0}
GPU=${3:-0}
ITERS=${4:-100000}
DATASET=${5:-/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda}
PORT=$((29600 + GPU))

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

RUN="${TASK}_tok_${EXP}"
mkdir -p train_logs
echo "=== $RUN  (action=$ACT proprio=$PROP $EXTRA)  gpu=$GPU iters=$ITERS ==="
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node 1 --master_port $PORT \
  main_trajectory_mimicgen.py \
  --tasks "$TASK" --dataset "$DATASET" \
  --batch_size 16 --batch_size_val 8 \
  --train_iters "$ITERS" --val_freq 2500 --val_iters 25 --num_workers 6 \
  --diffuse_gripper 1 \
  --action_token_groups "$ACT" --proprio_token_groups "$PROP" \
  $EXTRA \
  --rollout_freq 0 \
  --run_log_dir "$RUN" \
  2>&1 | tee "train_logs/${RUN}.log"
