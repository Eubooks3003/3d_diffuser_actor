# Remote training — mimicgen tokenization sweep (no rollouts)

Training-only setup for the lambda remote. **No robotics/sim stack** and **no
in-loop rollouts** (`--rollout_freq 0`) — the remote just trains and logs
train/val loss. Rollout evaluation is done separately on a machine with a sim.

## 1. Clone the fork branch
```bash
git clone -b mimicgen-tokenization git@github.com:Eubooks3003/3d_diffuser_actor.git
cd 3d_diffuser_actor
```

## 2. Create the training env
```bash
bash setup_remote_train_env.sh 3dda_train
conda activate 3dda_train
# sanity: model imports without the sim stack
python -c "from diffuser_actor import DiffuserActor; print('import OK')"
```
Installs (in order): torch 2.4.1+cu121 → flash-attn → `requirements_train.txt`.
flash-attn needs an Ampere+ GPU (A100/H100 are fine).

## 3. Data
Packed `.dat` (already blosc-compressed, RGB+depth+camera matrices per frame)
live at:
```
/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda/<task>/{ep*.dat, meta.json, episode_lengths.json}
```
One packing serves both absolute and relative policies. Single-task sweep needs
only `stack_d0/` (~0.9 GB); multitask needs all 12 (~34 GB).

## 4. Run
Point `--dataset` at the NFS copy and disable rollouts. Example (baseline):
```bash
torchrun --nproc_per_node 1 --master_port 29600 main_trajectory_mimicgen.py \
  --tasks stack_d0 \
  --dataset /lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda \
  --batch_size 16 --batch_size_val 8 \
  --train_iters 100000 --val_freq 2500 --val_iters 25 --num_workers 6 \
  --rollout_freq 0 \
  --action_token_groups '[3,6,1]' --proprio_token_groups '[3,6,1]' \
  --run_log_dir stack_d0_baseline
```

The 7 tokenization experiments are driven by
`scripts/train_mimicgen_tokenization_sweep.sh` (see that file for the exact
per-experiment `--action_token_groups` / `--proprio_token_groups`).

## No-eval sanity signal
Since rollouts are off, use `scripts/check_learning.py` to confirm the model is
actually learning (beats repeat-current-pose / random-pose baselines) without a
simulator.
```bash
python scripts/check_learning.py --checkpoint train_logs/mimicgen/<run>/last.pth \
  --dataset /lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_3dda --task stack_d0
```
