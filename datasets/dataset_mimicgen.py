"""Dataset for mimicgen episodes packaged by data_preprocessing/package_mimicgen.py.

Unlike the RLBench dataset, mimicgen has no keyposes: this serves dense action
chunks (horizon future EEF poses) for the trajectory head, matching the
horizon=16 setup EC-Diffuser used on the same trajectories.
"""
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import blosc
import numpy as np
import torch
from torch.utils.data import Dataset

from diffuser_actor.utils.utils import matrix_to_quaternion


def _load_episode(path):
    with open(path, "rb") as f:
        return pickle.loads(blosc.decompress(f.read()))


class MimicgenDataset(Dataset):
    """Dense-trajectory mimicgen dataset.

    Poses are (pos3, quat4 wxyz, openness1). Openness is derived from the
    COMMANDED gripper action, not from `gripper_state[..., 9]`: that channel is
    effectively constant (92% of values in one bin, correlation -0.075 with the
    gripper command) and lies outside [0, 1], so it is useless as a BCE target.
    """

    def __init__(
        self,
        root,
        tasks,
        horizon=16,
        nhist=3,
        training=True,
        cache_size=50,
        max_episodes_per_task=None,
        val_fraction=0.05,
        workspace_margin=0.25,
        relative_action=False,
        goal_actions=False,
        osc_pos_scale=0.05,   # robosuite OSC_POSE output_max (verified from env hdf5s)
        osc_rot_scale=0.5,
    ):
        self._root = Path(root)
        self._tasks = list(tasks)
        self._horizon = horizon
        self._nhist = nhist
        self._training = training
        self._cache_size = cache_size
        self._relative_action = relative_action
        self._goal_actions = goal_actions
        self._osc_pos_scale = osc_pos_scale
        self._osc_rot_scale = osc_rot_scale
        self._cache = OrderedDict()

        self.task_to_id = {t: i for i, t in enumerate(self._tasks)}
        self._meta, self._pix_to_world, self._grids = {}, {}, {}
        self._index = []  # (task, episode_path, t)

        loc_min, loc_max = [], []
        for task in self._tasks:
            tdir = self._root / task
            with open(tdir / "meta.json") as f:
                meta = json.load(f)
            self._meta[task] = meta
            self._cameras = meta["cameras"]
            size = meta["image_size"]
            self._pix_to_world[task] = np.stack([
                np.array(meta["cameras_matrices"][c]["pix_to_world"])
                for c in meta["cameras"]
            ])
            rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            self._grids[task] = (rows.astype(np.float64), cols.astype(np.float64))
            loc_min.append(meta["gripper_loc_bounds"][0])
            loc_max.append(meta["gripper_loc_bounds"][1])

            eps = sorted(tdir.glob("ep*.dat"))
            if max_episodes_per_task is not None:
                eps = eps[:max_episodes_per_task]
            # deterministic train/val split by episode, so val episodes are unseen
            n_val = max(1, int(round(len(eps) * val_fraction)))
            eps = eps[:-n_val] if training else eps[-n_val:]

            # Episode lengths are cached: decompressing every episode just to
            # count frames costs ~30s per dataset construction otherwise.
            lengths = self._episode_lengths(tdir, eps)
            for ep in eps:
                for t in range(lengths[ep.name]):
                    self._index.append((task, ep, t))

        # union of per-task bounds, so pos normalization is shared across tasks
        self.gripper_loc_bounds = np.stack([
            np.array(loc_min).min(0), np.array(loc_max).max(0)
        ])
        if self._goal_actions:
            # commanded goals can exceed achieved-pose bounds by one OSC step
            self.gripper_loc_bounds[0] -= self._osc_pos_scale
            self.gripper_loc_bounds[1] += self._osc_pos_scale

        # Raw MuJoCo depth includes the floor and skybox (points out to y=-3),
        # while the gripper spans only ~0.2m. Normalizing those by the gripper
        # bounds sends background coordinates to ~±20 and destroys the rotary
        # position encoding, so clamp the cloud to the workspace.
        self.workspace_bounds = np.stack([
            self.gripper_loc_bounds[0] - workspace_margin,
            self.gripper_loc_bounds[1] + workspace_margin,
        ])

        if self._relative_action:
            # In relative mode the model recentres the scene on the current
            # gripper (`convert2rel`), so the SAME bounds normalize both the
            # gripper-relative point cloud and the displacement targets. Reusing
            # the absolute bounds here would be catastrophic and silent: a zero
            # displacement in z would normalize to ~-1, because absolute z
            # bounds start at ~0.81. CALVIN ships a separate
            # `calvin_rel_traj_location_bounds` file for exactly this reason.
            self.gripper_loc_bounds = np.stack([
                self.workspace_bounds[0] - self.gripper_loc_bounds[1],
                self.workspace_bounds[1] - self.gripper_loc_bounds[0],
            ])

    @staticmethod
    def _episode_lengths(tdir, eps):
        """{filename: n_frames}, memoized in the task dir."""
        cache_path = tdir / "episode_lengths.json"
        cache = {}
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)
        missing = [e for e in eps if e.name not in cache]
        for ep in missing:
            cache[ep.name] = len(_load_episode(ep)["action"])
        if missing:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        return cache

    def _get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        data = _load_episode(path)
        self._cache[path] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def _poses(self, data, goal=False):
        """(T, 8) pos3 + quat4 (wxyz) + openness1.

        goal=False: ACHIEVED poses (where the demo hand was) — used for the
        proprio history, which at rollout comes from the sim's achieved state.
        goal=True: COMMANDED goal poses implied by the demo's delta action
        (goal = achieved (+) action * osc_scale) — the robosuite OSC target.
        Carries the demonstrator's force intent through contact (pressing
        'through' a surface), which achieved poses strip out; informationally
        the same signal EC-Diffuser trains on (raw deltas), re-expressed as
        absolute poses (Diffusion Policy's robomimic convention). Used for the
        FUTURE TARGETS only — mixing it into proprio would train on a state
        distribution the rollout never produces.
        """
        pos = data["eef_pos_replay"].astype(np.float64)
        mat = data["eef_mat_replay"].astype(np.float64).reshape(-1, 3, 3)
        if goal:
            from scipy.spatial.transform import Rotation
            act = data["action"].astype(np.float64)
            pos = pos + act[:, :3] * self._osc_pos_scale
            d_rot = Rotation.from_rotvec(act[:, 3:6] * self._osc_rot_scale)
            mat = np.einsum("tij,tjk->tik", d_rot.as_matrix(), mat)
        pos = torch.from_numpy(pos).float()
        quat = matrix_to_quaternion(torch.from_numpy(mat).float())
        # robosuite OSC: gripper action > 0 closes, < 0 opens
        openness = (torch.from_numpy(data["action"][:, 6]).float() < 0).float()
        return torch.cat([pos, quat, openness[:, None]], dim=-1)

    def _pcd(self, task, depth):
        """(ncam, H, W) metric depth -> (ncam, 3, H, W) world points."""
        rows, cols = self._grids[task]
        clouds = []
        for ci in range(depth.shape[0]):
            z = depth[ci].astype(np.float64)
            cam_pts = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)
            pts = cam_pts @ self._pix_to_world[task][ci].T
            clouds.append(pts[..., :3].transpose(2, 0, 1))
        cloud = np.stack(clouds)
        lo = self.workspace_bounds[0][None, :, None, None]
        hi = self.workspace_bounds[1][None, :, None, None]
        return torch.from_numpy(np.clip(cloud, lo, hi)).float()

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        task, ep_path, t = self._index[idx]
        data = self._get(ep_path)
        T = len(data["action"])
        # targets: goal poses (if enabled); proprio history: ALWAYS achieved
        # poses, matching what the rollout reads from the sim.
        tgt_poses = self._poses(data, goal=self._goal_actions)
        hist_poses = (self._poses(data, goal=False)
                      if self._goal_actions else tgt_poses)

        # future chunk, padded by repeating the final pose (the episode has
        # ended, so "hold still" is the semantically correct target)
        fut = [tgt_poses[min(t + 1 + i, T - 1)] for i in range(self._horizon)]
        trajectory = torch.stack(fut)

        # history, padded at the start by repeating the first pose
        hist = [hist_poses[max(t - self._nhist + 1 + j, 0)]
                for j in range(self._nhist)]
        curr_gripper = torch.stack(hist)

        if self._relative_action:
            # Every horizon step is expressed relative to the CURRENT pose,
            # matching CALVIN (`to_relative_action(traj[j], traj[0])`) and the
            # model's `convert2rel`, which recentres pcd + curr_gripper on the
            # same point. Openness stays absolute.
            trajectory = self._to_relative(trajectory, curr_gripper[-1])

        rgb = torch.from_numpy(data["rgb"][t]).float().permute(0, 3, 1, 2) / 255.0
        pcd = self._pcd(task, data["depth"][t].astype(np.float32))

        return {
            "rgb": rgb,                      # (ncam, 3, H, W) in [0, 1]
            "pcd": pcd,                      # (ncam, 3, H, W) world coords
            "trajectory": trajectory,        # (horizon, 8)
            "trajectory_mask": torch.zeros(self._horizon, dtype=torch.bool),
            "curr_gripper": curr_gripper,    # (nhist, 8)
            "task_id": torch.tensor(self.task_to_id[task], dtype=torch.long),
        }

    @staticmethod
    def _to_relative(traj, base):
        """Express `traj` poses relative to `base` (pos delta + quat delta).

        CALVIN's `to_relative_action` uses Euler angles; our poses are
        quaternions, so the rotation delta is q_rel = q_target * q_base^-1.
        """
        out = traj.clone()
        out[:, :3] = traj[:, :3] - base[:3]

        bq = base[3:7]
        binv = torch.cat([bq[:1], -bq[1:]]) / (bq.dot(bq) + 1e-8)  # wxyz inverse
        w1, x1, y1, z1 = traj[:, 3], traj[:, 4], traj[:, 5], traj[:, 6]
        w2, x2, y2, z2 = binv[0], binv[1], binv[2], binv[3]
        out[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out[:, 4] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out[:, 5] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out[:, 6] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return out
