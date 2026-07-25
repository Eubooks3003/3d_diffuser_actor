"""Evaluate ONE multitask checkpoint on ONE task: seeds x n_rollouts, fresh
init states. Writes a JSON with per-seed success rates. Meant to be launched by
scripts/eval_queue.py (one process per (experiment, task) job), but runnable
standalone.

    python online_evaluation_mimicgen/eval_multitask_worker.py \
        --checkpoint remote_ckpts/multitask_tok_baseline_best.pth \
        --experiment baseline --task stack_d0 \
        --seeds 42,123,456 --n_rollouts 50 --output out.json
"""
import argparse, json, os, random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from online_evaluation_mimicgen.eval_mimicgen import (
    build_env, controller_scales, load_meta, rollout,
    DEFAULT_ENV_ROOT, DEFAULT_PACKED,
)
from diffuser_actor import DiffuserActor

# Must match the training task order (train_one_experiment.sh ALL12) so task_id
# lines up with the task_embedding the checkpoint learned.
ALL12 = ["coffee_d0", "coffee_preparation_d0", "hammer_cleanup_d0", "kitchen_d0",
         "mug_cleanup_d0", "nut_assembly_d0", "pick_place_d0", "square_d0",
         "stack_d0", "stack_three_d0", "threading_d0", "three_piece_assembly_d0"]

# experiment -> (action_token_groups, proprio_token_groups, no_proprio)
CONFIGS = {
    "baseline":       ([3, 6, 1],    [3, 6, 1],    False),
    "single_action":  ([10],         [3, 6, 1],    False),
    "uniform":        ("per_dim",    "per_dim",    False),
    "random1":        ([2, 1, 4, 3], [4, 1, 2, 3], False),
    "random2":        ([3, 1, 2, 4], [2, 3, 1, 4], False),
    "no_proprio":     ([3, 6, 1],    None,         True),
    "single_proprio": ([3, 6, 1],    [10],         False),
}


def union_bounds(packed_root, tasks):
    mn, mx = [], []
    for t in tasks:
        b = np.array(load_meta(packed_root, t)["gripper_loc_bounds"])
        mn.append(b[0]); mx.append(b[1])
    return np.stack([np.min(mn, 0), np.max(mx, 0)])


def build_model(experiment, checkpoint, bounds, device):
    atg, ptg, no_prop = CONFIGS[experiment]
    model = DiffuserActor(
        backbone="clip", image_size=(128, 128), embedding_dim=120,
        use_instruction=False, fps_subsampling_factor=5, gripper_loc_bounds=bounds,
        rotation_parametrization="6D", quaternion_format="wxyz",
        diffusion_timesteps=100, nhist=3, relative=False,
        action_token_groups=atg, proprio_token_groups=ptg,
        no_proprio=no_prop, diffuse_gripper=True, n_tasks=len(ALL12),
    ).to(device)
    ck = torch.load(checkpoint, map_location="cpu")
    state = ck.get("weight", ck.get("model", ck))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    if miss or unexp:
        print(f"WARNING load: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--experiment", required=True, choices=list(CONFIGS))
    p.add_argument("--task", required=True, choices=ALL12)
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--output", required=True)
    p.add_argument("--packed_root", default=DEFAULT_PACKED)
    p.add_argument("--env_root", default=DEFAULT_ENV_ROOT)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    bounds = union_bounds(args.packed_root, ALL12)
    model = build_model(args.experiment, args.checkpoint, bounds, args.device)

    meta = load_meta(args.packed_root, args.task)
    size = meta["image_size"]
    env, env_meta = build_env(args.env_root, args.task, "absolute")
    pos_scale, rot_scale = controller_scales(env_meta)
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    b = bounds
    cfg = dict(
        cameras=meta["cameras"], size=size, nhist=3, horizon=16, exe_steps=8,
        max_steps=args.max_steps,
        grid=(rows.astype(np.float64), cols.astype(np.float64)),
        workspace=np.stack([b[0] - 0.25, b[1] + 0.25]),
        pos_scale=pos_scale, rot_scale=rot_scale, control_mode="absolute",
        fresh_reset=True,  # sample a new init from the task distribution each rollout
        task_id_t=torch.tensor([ALL12.index(args.task)], device=args.device),
    )

    per_seed = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        try:
            env.env.seed(seed)
        except Exception:
            pass
        succ = 0
        for i in range(args.n_rollouts):
            ok = rollout(model, env, None, 0, cfg, args.device)
            succ += int(bool(ok))
        rate = succ / args.n_rollouts
        per_seed.append({"seed": seed, "success_rate": rate,
                         "successes": succ, "n": args.n_rollouts})
        print(f"[{args.experiment}/{args.task}] seed {seed}: "
              f"{succ}/{args.n_rollouts} = {rate:.3f}", flush=True)

    rates = [r["success_rate"] for r in per_seed]
    out = {
        "experiment": args.experiment, "task": args.task,
        "n_rollouts": args.n_rollouts, "seeds": seeds,
        "per_seed_results": per_seed,
        "mean": float(np.mean(rates)), "std": float(np.std(rates)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.experiment}/{args.task}] mean={out['mean']:.3f} "
          f"std={out['std']:.3f} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
