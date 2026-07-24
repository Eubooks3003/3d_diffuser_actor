"""Online rollout evaluation of 3D Diffuser Actor on mimicgen.

The policy predicts ABSOLUTE end-effector poses, but mimicgen runs a robosuite
OSC_POSE controller in delta mode, so each predicted pose is converted into a
normalized delta command before stepping. That conversion is where success rate
is most easily lost, so the scaling is taken from the env's own controller
config rather than hard-coded.

Usage:
    python eval_mimicgen.py --checkpoint path/to/best.pth --task stack_d0 \
        --num-episodes 25
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import torch  # noqa: E402
import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
from robosuite.utils import camera_utils as CamUtils  # noqa: E402
from robosuite.utils.transform_utils import (  # noqa: E402
    mat2quat, quat2mat, quat2axisangle, quat_multiply, quat_inverse,
)

from diffuser_actor import DiffuserActor  # noqa: E402

DEFAULT_PKL_ROOT = "/home/ellina/Desktop/data/preprocessed_multiview_tokens"
DEFAULT_ENV_ROOT = "/home/ellina/Desktop/data/3D-DLP-mimicgen-data/core"
DEFAULT_PACKED = "/home/ellina/Desktop/data/mimicgen_3dda"


def load_meta(packed_root, task):
    with open(Path(packed_root) / task / "meta.json") as f:
        return json.load(f)


def load_gt_poses(packed_root, task, ep):
    """(T, 8) ground-truth pos + quat(wxyz) + openness for the oracle check."""
    import blosc
    with open(Path(packed_root) / task / f"ep{ep:04d}.dat", "rb") as f:
        d = pickle.loads(blosc.decompress(f.read()))
    pos = d["eef_pos_replay"]
    mats = d["eef_mat_replay"].reshape(-1, 3, 3)
    quats = []
    for m in mats:
        q_xyzw = mat2quat(m)
        quats.append(np.concatenate([q_xyzw[3:], q_xyzw[:3]]))
    openness = (d["action"][:, 6] < 0).astype(np.float32)[:, None]
    return np.concatenate([pos, np.stack(quats), openness], axis=-1)


def build_env(env_root, task, control_mode="delta"):
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_eef_pos"]}}
    )
    with h5py.File(Path(env_root) / f"{task}.hdf5", "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    if control_mode == "absolute":
        # The policy predicts absolute EEF poses. Chasing them with recomputed
        # per-step deltas makes the OSC controller lag behind a moving
        # reference, so command the pose directly instead.
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True
    )
    return env, env_meta


def controller_scales(env_meta):
    """(pos_scale, rot_scale) mapping a normalized action to a metric delta."""
    cfg = env_meta["env_kwargs"]["controller_configs"]
    out_max = np.array(cfg["output_max"], dtype=np.float64)
    return out_max[:3], out_max[3:6]


def render_obs(sim, cameras, size):
    """RGB in [0,1] and metric depth, in top-down ordering (see packager)."""
    rgbs, depths = [], []
    for cam in cameras:
        rgb, depth = sim.render(width=size, height=size, camera_name=cam, depth=True)
        depth = CamUtils.get_real_depth_map(sim=sim, depth_map=depth[..., None])
        rgbs.append(rgb[::-1].copy())
        depths.append(depth[::-1, :, 0].copy())
    return np.stack(rgbs), np.stack(depths)


def make_pcd(depth, pix_to_world, grid, workspace):
    rows, cols = grid
    clouds = []
    for ci in range(depth.shape[0]):
        z = depth[ci].astype(np.float64)
        cam_pts = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)
        pts = cam_pts @ pix_to_world[ci].T
        clouds.append(pts[..., :3].transpose(2, 0, 1))
    cloud = np.stack(clouds)
    lo = workspace[0][None, :, None, None]
    hi = workspace[1][None, :, None, None]
    return np.clip(cloud, lo, hi)


def quat_mul_wxyz(a, b):
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def eef_pose(sim):
    """(pos3, quat4 wxyz) of the grip site."""
    sid = sim.model.site_name2id("gripper0_grip_site")
    pos = np.array(sim.data.site_xpos[sid])
    mat = np.array(sim.data.site_xmat[sid]).reshape(3, 3)
    quat_xyzw = mat2quat(mat)
    quat_wxyz = np.concatenate([quat_xyzw[3:], quat_xyzw[:3]])
    return pos, quat_wxyz


def pose_to_action(cur_pos, cur_quat_wxyz, tgt_pos, tgt_quat_wxyz,
                   openness, pos_scale, rot_scale, control_mode="delta"):
    """Absolute target pose -> OSC action, in delta or absolute mode."""
    # robosuite gripper: +1 closes, -1 opens; `openness` is 1 when open
    grip = -1.0 if openness > 0.5 else 1.0

    if control_mode == "absolute":
        # use_delta=False: action is [target_pos, target_axisangle], unscaled
        tgt_xyzw = np.concatenate([tgt_quat_wxyz[1:], tgt_quat_wxyz[:1]])
        return np.concatenate([tgt_pos, quat2axisangle(tgt_xyzw), [grip]])

    dpos = np.clip((tgt_pos - cur_pos) / pos_scale, -1.0, 1.0)

    cur_xyzw = np.concatenate([cur_quat_wxyz[1:], cur_quat_wxyz[:1]])
    tgt_xyzw = np.concatenate([tgt_quat_wxyz[1:], tgt_quat_wxyz[:1]])
    # relative rotation: tgt = delta * cur  ->  delta = tgt * cur^-1
    dquat = quat_multiply(tgt_xyzw, quat_inverse(cur_xyzw))
    daa = quat2axisangle(dquat)
    drot = np.clip(daa / rot_scale, -1.0, 1.0)
    return np.concatenate([dpos, drot, [grip]])


@torch.no_grad()
def rollout(model, env, data, ep, cfg, device, collect_frames=False):
    env.reset()
    env.reset_to({"states": data["init_states"][ep]})
    # robosuite rebuilds the sim on reset, so this must be re-fetched per episode
    sim = env.env.sim

    mats = np.stack([
        np.linalg.inv(CamUtils.get_camera_transform_matrix(
            sim=sim, camera_name=c, camera_height=cfg["size"], camera_width=cfg["size"]))
        for c in cfg["cameras"]
    ])

    pos, quat = eef_pose(sim)
    history = [np.concatenate([pos, quat, [1.0]])] * cfg["nhist"]

    frames = []
    success = False
    for step in range(0, cfg["max_steps"], cfg["exe_steps"]):
        if cfg.get("oracle") is not None:
            # Ground-truth poses pushed through the SAME pose->action
            # conversion. This separates "the controller conversion and success
            # detection work" from "the policy is undertrained": a 0% policy
            # result is only meaningful if the oracle scores high here.
            gt = cfg["oracle"]
            traj = gt[min(step + 1, len(gt) - 1):step + 1 + cfg["horizon"]]
            if len(traj) == 0:
                break
        else:
            rgb, depth = render_obs(sim, cfg["cameras"], cfg["size"])
            pcd = make_pcd(depth, mats, cfg["grid"], cfg["workspace"])

            rgb_t = torch.from_numpy(rgb).float().permute(0, 3, 1, 2)[None] / 255.0
            pcd_t = torch.from_numpy(pcd).float()[None]
            cg = torch.from_numpy(np.stack(history[-cfg["nhist"]:])).float()[None]
            mask = torch.zeros(1, cfg["horizon"], dtype=torch.bool)

            traj = model(
                None, mask.to(device), rgb_t.to(device), pcd_t.to(device),
                None, cg.to(device), run_inference=True
            )[0].cpu().numpy()  # (horizon, 8) pos + quat(wxyz) + openness

        if cfg.get("relative_action") and cfg.get("oracle") is None:
            # The policy predicts displacements from the pose at prediction
            # time; convert back to absolute targets so the (oracle-validated)
            # absolute control path is unchanged.
            b_pos, b_quat = eef_pose(sim)
            traj = traj.copy()
            traj[:, :3] = traj[:, :3] + b_pos
            traj[:, 3:7] = np.stack([
                quat_mul_wxyz(q, b_quat) for q in traj[:, 3:7]
            ])

        for i in range(min(cfg["exe_steps"], len(traj))):
            cur_pos, cur_quat = eef_pose(sim)
            tgt = traj[i]
            tgt_quat = tgt[3:7] / (np.linalg.norm(tgt[3:7]) + 1e-8)
            action = pose_to_action(
                cur_pos, cur_quat, tgt[:3], tgt_quat, tgt[7],
                cfg["pos_scale"], cfg["rot_scale"], cfg["control_mode"]
            )
            env.step(action)
            if collect_frames:
                frames.append(
                    sim.render(width=cfg["size"], height=cfg["size"],
                               camera_name=cfg["cameras"][0])[::-1].copy()
                )
            p, q = eef_pose(sim)
            history.append(np.concatenate([p, q, [tgt[7]]]))
            if env.is_success()["task"]:
                success = True
                break
        if success:
            break
    return (success, frames) if collect_frames else success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="stack_d0")
    p.add_argument("--num-episodes", type=int, default=25)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--exe-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--nhist", type=int, default=3)
    p.add_argument("--embedding-dim", type=int, default=120)
    p.add_argument("--action-token-groups", default="default")
    p.add_argument("--diffusion-timesteps", type=int, default=100)
    p.add_argument("--packed-root", default=DEFAULT_PACKED)
    p.add_argument("--pkl-root", default=DEFAULT_PKL_ROOT)
    p.add_argument("--env-root", default=DEFAULT_ENV_ROOT)
    # evaluate on the episodes the dataset held out (its last 5%)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--control-mode", choices=["delta", "absolute"],
                   default="absolute")
    p.add_argument("--relative-action", action="store_true",
                   help="policy predicts displacements (must match training)")
    p.add_argument("--oracle", action="store_true",
                   help="replay ground-truth poses through the pose->action "
                        "conversion instead of the policy (harness sanity check)")
    args = p.parse_args()

    device = "cuda"
    meta = load_meta(args.packed_root, args.task)
    size = meta["image_size"]
    cameras = meta["cameras"]
    bounds = np.array(meta["gripper_loc_bounds"])
    workspace = np.stack([bounds[0] - 0.25, bounds[1] + 0.25])
    if args.relative_action:
        # Must match MimicgenDataset exactly: in relative mode the model
        # normalizes displacements (and the recentred cloud), not absolute
        # positions. Using absolute bounds here would silently wreck inference.
        bounds = np.stack([workspace[0] - bounds[1], workspace[1] - bounds[0]])

    with open(Path(args.pkl_root) / args.task / f"{args.task}.pkl", "rb") as f:
        data = pickle.load(f)

    n_total = len(data["path_lengths"])
    n_val = max(1, int(round(min(n_total, 200) * args.val_fraction)))
    val_eps = list(range(min(n_total, 200) - n_val, min(n_total, 200)))
    eps = (val_eps * ((args.num_episodes // len(val_eps)) + 1))[:args.num_episodes]
    print(f"held-out episodes: {val_eps} -> evaluating {len(eps)} rollouts")

    model = DiffuserActor(
        backbone="clip", image_size=(size, size), embedding_dim=args.embedding_dim,
        use_instruction=False, fps_subsampling_factor=5,
        gripper_loc_bounds=bounds, rotation_parametrization="6D",
        quaternion_format="wxyz", diffusion_timesteps=args.diffusion_timesteps,
        nhist=args.nhist, action_token_groups=args.action_token_groups,
        relative=args.relative_action,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("weight", ckpt.get("model", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARNING missing={len(missing)} unexpected={len(unexpected)} keys")
    model.eval()

    env, env_meta = build_env(args.env_root, args.task, args.control_mode)
    pos_scale, rot_scale = controller_scales(env_meta)
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    cfg = dict(
        cameras=cameras, size=size, nhist=args.nhist, horizon=args.horizon,
        exe_steps=args.exe_steps, max_steps=args.max_steps,
        grid=(rows.astype(np.float64), cols.astype(np.float64)),
        workspace=workspace, pos_scale=pos_scale, rot_scale=rot_scale,
        control_mode=args.control_mode,
        relative_action=args.relative_action,
    )

    successes = []
    for i, ep in enumerate(eps):
        if args.oracle:
            cfg["oracle"] = load_gt_poses(args.packed_root, args.task, ep)
        ok = rollout(model, env, data, ep, cfg, device)
        successes.append(ok)
        print(f"  [{i+1}/{len(eps)}] episode {ep}: {'SUCCESS' if ok else 'fail'}"
              f"   running SR = {np.mean(successes):.3f}")

    print(f"\n{args.task}: success rate {np.mean(successes):.3f} "
          f"({int(np.sum(successes))}/{len(successes)})")


if __name__ == "__main__":
    main()
