# Remote training — mimicgen tokenization sweep (no rollouts)

Training-only setup. **No robotics/sim stack** and **no in-loop rollouts**
(`--rollout_freq 0`) — the remote just trains and logs train/val loss. Rollout
evaluation is done separately on a machine with a simulator.

## Per-server setup (run on each server)

`/lambda/nfs` is shared across servers, so the repo + data are already visible
everywhere — each server only needs its own local conda env.

```bash
cd /path/to/3d_diffuser_actor          # shared checkout on NFS
git pull                               # get the latest
bash setup_remote_train_env.sh 3dda_train    # idempotent + self-verifying
conda activate 3dda_train
```
The script is arch-aware (x86_64 → cu121, aarch64/GH200 → cu124), upgrades pip
first, and **skips flash-attn** (the attention layers fall back to torch SDPA,
which runs FlashAttention-2 kernels on Hopper). It prints `import OK` when done.
To also build flash-attn: `INSTALL_FLASH=1 bash setup_remote_train_env.sh`.

## Data
Packed `.dat` (already blosc-compressed) at:
```
/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda/<task>/{ep*.dat, meta.json, episode_lengths.json}
```
One packing serves absolute and relative policies. `stack_d0/` alone (~0.9 GB)
suffices for the single-task sweep.

## Run — one experiment per server

Each server trains one token config with `train_one_experiment.sh`. The 2nd arg
is the task set: a single task, a quoted list, or `all12` for the **multitask**
policy over all 12 tasks (task-ID embedding turns on automatically; token groups
are identical). Multitask needs all 12 tasks packed on the NFS (~34 GB).
```bash
# experiment in: baseline single_action uniform random1 random2 no_proprio single_proprio

# single task:
bash scripts/train_one_experiment.sh <experiment> stack_d0 0 100000 \
  /lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda

# multitask (all 12 tasks, one policy) — use a larger iter budget (see below):
bash scripts/train_one_experiment.sh <experiment> all12 0 300000 \
  /lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda
```
Iterations: single-task stack_d0 converges by ~20k (100k is ample). Multitask
(12 tasks, ~10x the data, plus task disambiguation) is harder — use ~200-300k,
watch val-loss for plateau/overfit, and compare on `best.pth`. Not 600k; that
was RLBench-18 + language, heavier than mimicgen multitask.
Suggested assignment (7 experiments, 4 servers — double up two servers):

| server | experiment(s) |
|--------|---------------|
| 1 | `baseline` |
| 2 | `single_action` |
| 3 | `uniform`, then `random1` |
| 4 | `single_proprio`, then `random2`, then `no_proprio` |

Logs → `train_logs/stack_d0_tok_<experiment>.log`; checkpoints →
`train_logs/mimicgen/stack_d0_tok_<experiment>/{best,last}.pth`.

All 7 on one server (sequential): `bash scripts/train_mimicgen_tokenization_sweep.sh`.

The token groups (10-dim = pos(3)/rot6d(6)/gripper(1)) are defined once in
`scripts/train_one_experiment.sh`.

## No-eval sanity signal
Rollouts are off, so confirm learning with `check_learning.py` (beats
repeat-current-pose / random-pose baselines; no simulator):
```bash
python scripts/check_learning.py \
  --checkpoint train_logs/mimicgen/stack_d0_tok_baseline/last.pth \
  --dataset /lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda --task stack_d0
```
