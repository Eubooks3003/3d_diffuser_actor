"""Train 3D Diffuser Actor's trajectory head on mimicgen.

Differences from main_trajectory.py (RLBench/CALVIN):
  * dense action chunks instead of keyposes (mimicgen is 20Hz OSC control)
  * NO language conditioning. Multitask conditioning is a learned task-ID
    embedding added to the diffusion timestep embedding, mirroring
    EC-Diffuser's `pint.py`, so the comparison isn't confounded by a different
    conditioning pathway.
"""
import os
import pickle
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tap
import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate

from datasets.dataset_mimicgen import MimicgenDataset
from diffuser_actor import DiffuserActor
from engine import BaseTrainTester
from utils.common_utils import count_parameters


class Arguments(tap.Tap):
    dataset: Path = Path("/home/ellina/Desktop/data/mimicgen_3dda")
    tasks: Tuple[str, ...] = ("stack_d0",)
    image_size: str = "128,128"
    seed: int = 0
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    eval_only: int = 0
    max_episodes_per_task: Optional[int] = None

    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "mimicgen"
    run_log_dir: str = "run"

    num_workers: int = 4
    batch_size: int = 16
    batch_size_val: int = 8
    cache_size: int = 30
    cache_size_val: int = 10
    lr: float = 1e-4
    wd: float = 5e-3
    train_iters: int = 200_000
    val_iters: int = 50

    # model
    backbone: str = "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    fps_subsampling_factor: int = 5
    rotation_parametrization: str = "6D"
    # our dataset emits wxyz quaternions, so no permutation is needed
    quaternion_format: str = "wxyz"
    diffusion_timesteps: int = 100
    num_history: int = 3
    relative_action: int = 0
    lang_enhanced: int = 0

    # trajectory
    horizon: int = 16

    # tokenization (Phase 4); 'per_dim' == EC-Diffuser's "uniform action"
    action_token_groups: str = "default"
    proprio_token_groups: str = "default"
    no_proprio: int = 0  # EC-Diffuser "no proprio" ablation; requires absolute actions

    # In-training rollout eval: actually runs the policy in the simulator and
    # reports task success, which loss/pos_err cannot tell you.
    rollout_freq: int = 10000        # in steps; 0 disables
    rollout_episodes: int = 5        # per task
    rollout_max_steps: int = 300
    rollout_control_mode: str = "absolute"   # see eval_mimicgen: delta lags
    rollout_video: int = 1           # save an mp4 of the first rollout
    # 'train' rolls out on TRAINING episodes -- use with a small
    # --max_episodes_per_task as a fast "is it learning at all?" check.
    rollout_split: str = "val"


def parse_groups(spec):
    """CLI strings -> token-group spec ('default', 'per_dim', or '[3,6]')."""
    if spec in ('default', 'per_dim'):
        return spec
    return [int(x) for x in spec.strip('[]').replace(' ', '').split(',') if x]


class TrainTester(BaseTrainTester):

    def __init__(self, args):
        super().__init__(args)
        # robosuite envs are expensive to build and leak GL contexts if rebuilt
        # every eval, so they are created once and reused.
        self._rollout_envs = {}
        self._rollout_data = {}

    def _rollout_cfg(self, task):
        """Build (env, cfg, data) for a task, cached across eval cycles."""
        from online_evaluation_mimicgen.eval_mimicgen import (
            build_env, controller_scales, load_meta, DEFAULT_ENV_ROOT,
            DEFAULT_PKL_ROOT,
        )
        if task not in self._rollout_envs:
            meta = load_meta(self.args.dataset, task)
            env, env_meta = build_env(
                DEFAULT_ENV_ROOT, task, self.args.rollout_control_mode
            )
            pos_scale, rot_scale = controller_scales(env_meta)
            size = meta["image_size"]
            rows, cols = np.meshgrid(
                np.arange(size), np.arange(size), indexing="ij"
            )
            bounds = np.array(meta["gripper_loc_bounds"])
            self._rollout_envs[task] = (env, dict(
                cameras=meta["cameras"], size=size, nhist=self.args.num_history,
                horizon=self.args.horizon, exe_steps=self.args.horizon // 2,
                max_steps=self.args.rollout_max_steps,
                grid=(rows.astype(np.float64), cols.astype(np.float64)),
                workspace=np.stack([bounds[0] - 0.25, bounds[1] + 0.25]),
                pos_scale=pos_scale, rot_scale=rot_scale,
                control_mode=self.args.rollout_control_mode,
                relative_action=bool(self.args.relative_action),
            ))
            with open(Path(DEFAULT_PKL_ROOT) / task / f"{task}.pkl", "rb") as f:
                self._rollout_data[task] = pickle.load(f)
        env, cfg = self._rollout_envs[task]
        return env, cfg, self._rollout_data[task]

    @torch.no_grad()
    def run_rollouts(self, model, step_id):
        """Run the policy in-sim on held-out episodes and log success rate."""
        from online_evaluation_mimicgen.eval_mimicgen import rollout

        device = next(model.parameters()).device
        net = model.module if hasattr(model, "module") else model
        net.eval()

        overall = []
        for task in self.args.tasks:
            env, cfg, data = self._rollout_cfg(task)
            # Must mirror MimicgenDataset's split exactly, INCLUDING
            # max_episodes_per_task -- otherwise an overfit run would roll out
            # on episodes that are not even in the dataset.
            n_eps = min(200, len(data["path_lengths"]))
            if self.args.max_episodes_per_task is not None:
                n_eps = min(n_eps, self.args.max_episodes_per_task)
            n_val = max(1, int(round(n_eps * 0.05)))
            if self.args.rollout_split == "train":
                eps_pool = list(range(0, n_eps - n_val))
            else:
                eps_pool = list(range(n_eps - n_val, n_eps))
            eps_pool = eps_pool[:self.args.rollout_episodes]

            succ, frames0 = [], None
            for j, ep in enumerate(eps_pool):
                want_video = bool(self.args.rollout_video) and j == 0
                out = rollout(net, env, data, ep, cfg, device,
                              collect_frames=want_video)
                ok, frames = out if want_video else (out, None)
                succ.append(bool(ok))
                if want_video and frames:
                    frames0 = frames

            sr = float(np.mean(succ)) if succ else 0.0
            overall.extend(succ)
            print(f"[step {step_id}] ROLLOUT[{self.args.rollout_split}] {task}: "
                  f"success {sr:.2f} ({int(np.sum(succ))}/{len(succ)})  "
                  f"eps={eps_pool}", flush=True)
            self.writer.add_scalar(f"rollout/{task}_success", sr, step_id)

            if frames0:
                import imageio
                base = Path(self.args.log_dir) / f"rollout_{task}_{step_id}"
                for ext, kw in ((".mp4", {"fps": 20}), (".gif", {"duration": 0.05})):
                    try:
                        imageio.mimsave(str(base) + ext, frames0, **kw)
                        print(f"           video -> {base}{ext}", flush=True)
                        break
                    except Exception as e:  # never fatal; gif is the fallback
                        print(f"           {ext} failed ({e}); trying next",
                              flush=True)

        if len(self.args.tasks) > 1:
            self.writer.add_scalar(
                "rollout/mean_success", float(np.mean(overall)), step_id
            )
        net.train()

    def get_datasets(self):
        train_dataset = MimicgenDataset(
            root=self.args.dataset,
            tasks=self.args.tasks,
            horizon=self.args.horizon,
            nhist=self.args.num_history,
            training=True,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            relative_action=bool(self.args.relative_action),
        )
        test_dataset = MimicgenDataset(
            root=self.args.dataset,
            tasks=self.args.tasks,
            horizon=self.args.horizon,
            nhist=self.args.num_history,
            training=False,
            cache_size=self.args.cache_size_val,
            max_episodes_per_task=self.args.max_episodes_per_task,
            relative_action=bool(self.args.relative_action),
        )
        # pos normalization must span both splits
        self.gripper_loc_bounds = train_dataset.gripper_loc_bounds
        print(f"Train frames: {len(train_dataset)}  Val frames: {len(test_dataset)}")
        print(f"gripper_loc_bounds:\n{self.gripper_loc_bounds}")
        return train_dataset, test_dataset

    def get_model(self):
        if self.args.no_proprio and self.args.relative_action:
            raise ValueError(
                "no_proprio requires absolute actions (relative_action=0): "
                "relative mode leaks the current pose back in via convert2rel"
            )
        _model = DiffuserActor(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=False,           # no language on mimicgen
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced),
            action_token_groups=parse_groups(self.args.action_token_groups),
            proprio_token_groups=(
                None if self.args.no_proprio
                else parse_groups(self.args.proprio_token_groups)
            ),
            no_proprio=bool(self.args.no_proprio),
            n_tasks=len(self.args.tasks),
        )
        print("Model parameters:", count_parameters(_model))
        return _model

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        device = next(model.parameters()).device
        out = model(
            sample["trajectory"].to(device),
            sample["trajectory_mask"].to(device),
            sample["rgb"].to(device),
            sample["pcd"].to(device),
            None,                                   # instruction
            sample["curr_gripper"].to(device),
            task_id=sample["task_id"].to(device),
        )
        loss = criterion.compute_loss(out)
        loss.backward()

        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split="val"):
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        device = next(model.parameters()).device
        model.eval()

        losses, pos_errs = [], []
        for i, sample in enumerate(loader):
            if i == val_iters:
                break
            gt = sample["trajectory"].to(device)
            tid = sample["task_id"].to(device)
            out = model(
                gt, sample["trajectory_mask"].to(device),
                sample["rgb"].to(device), sample["pcd"].to(device),
                None, sample["curr_gripper"].to(device), task_id=tid,
            )
            losses.append(criterion.compute_loss(out).item())

            pred = model(
                None, sample["trajectory_mask"].to(device),
                sample["rgb"].to(device), sample["pcd"].to(device),
                None, sample["curr_gripper"].to(device),
                run_inference=True, task_id=tid,
            )
            # position error in metres, the metric that actually matters
            pos_errs.append(
                (pred[..., :3] - gt[..., :3]).norm(dim=-1).mean().item()
            )

        values = {
            "loss": float(np.mean(losses)),
            "pos_err_m": float(np.mean(pos_errs)),
        }
        if dist.get_rank() == 0:
            for k, v in values.items():
                self.writer.add_scalar(f"{split}-loss/{k}", v, step_id)
            print(f"[step {step_id}] {split}: "
                  + "  ".join(f"{k}={v:.5f}" for k, v in values.items()),
                  flush=True)

            # Rollouts are the only signal that the policy does the TASK, not
            # just that it fits the trajectory distribution. Rank 0 only.
            if (split == "val" and self.args.rollout_freq > 0
                    and (step_id + 1) % self.args.rollout_freq == 0):
                self.run_rollouts(model, step_id)

        model.train()
        return values.get("pos_err_m")


class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, mask=None, is_loss=True):
        if not is_loss:
            assert gt is not None and mask is not None
            return self.compute_metrics(pred, gt, mask)[0]["action_mse"]
        return pred

    @staticmethod
    def compute_metrics(pred, gt, mask):
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        return {"action_mse": pos_l2.mean()}, pos_l2


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = Arguments().parse_args()

    args.log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir.mkdir(parents=True, exist_ok=True)
    print("Logging:", args.log_dir)
    args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(args)
    train_tester.main(collate_fn=default_collate)


if __name__ == "__main__":
    main()
