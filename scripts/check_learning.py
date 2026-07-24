"""Is the model learning anything? Compare it to trivial baselines.

Binary task success needs sub-cm precision and stays 0 for a long time, which
is indistinguishable from a broken pipeline. This is far more sensitive: if the
model beats "repeat the current pose" and "random pose", it has learned real
structure from the point cloud.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/ellina/Desktop/3d_diffuser_actor")
from datasets.dataset_mimicgen import MimicgenDataset
from diffuser_actor import DiffuserActor

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ellina/Desktop/3d_diffuser_actor/train_logs/mimicgen/stack_d0_default/last.pth"
N_BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 8

ds = MimicgenDataset(
    root="/home/ellina/Desktop/data/mimicgen_3dda", tasks=["stack_d0"],
    horizon=16, nhist=3, training=False, cache_size=10,
)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

ckpt = torch.load(CKPT, map_location="cpu")
print(f"checkpoint: {CKPT}\n  iter={ckpt.get('iter')}  best={ckpt.get('best_loss')}")
state = {k.replace("module.", ""): v for k, v in ckpt["weight"].items()}

model = DiffuserActor(
    backbone="clip", image_size=(128, 128), embedding_dim=120,
    use_instruction=False, fps_subsampling_factor=5,
    gripper_loc_bounds=ds.gripper_loc_bounds, rotation_parametrization="6D",
    quaternion_format="wxyz", diffusion_timesteps=100, nhist=3,
).cuda()
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"  missing={len(missing)} unexpected={len(unexpected)}")
model.eval()

lo, hi = ds.gripper_loc_bounds
rng = np.random.default_rng(0)
model_e, hold_e, rand_e = [], [], []

with torch.no_grad():
    for i, s in enumerate(loader):
        if i == N_BATCH:
            break
        gt = s["trajectory"].cuda()
        cg = s["curr_gripper"].cuda()
        pred = model(None, s["trajectory_mask"].cuda(), s["rgb"].cuda(),
                     s["pcd"].cuda(), None, cg, run_inference=True)
        model_e.append((pred[..., :3] - gt[..., :3]).norm(dim=-1).mean().item())

        # baseline 1: hold the current pose for the whole horizon
        hold = cg[:, -1:, :3].expand(-1, gt.shape[1], -1)
        hold_e.append((hold - gt[..., :3]).norm(dim=-1).mean().item())

        # baseline 2: uniform random pose inside the gripper workspace
        r = torch.from_numpy(
            rng.uniform(lo, hi, size=(gt.shape[0], gt.shape[1], 3))
        ).float().cuda()
        rand_e.append((r - gt[..., :3]).norm(dim=-1).mean().item())

m, h, rd = np.mean(model_e), np.mean(hold_e), np.mean(rand_e)
print(f"\nmean position error over {N_BATCH * 8} held-out samples:")
print(f"  MODEL                 {m:.4f} m")
print(f"  baseline: hold pose   {h:.4f} m   ({h / m:.2f}x worse)")
print(f"  baseline: random pose {rd:.4f} m   ({rd / m:.2f}x worse)")
verdict = "LEARNING (beats both baselines)" if m < h and m < rd else "NOT beating baselines"
print(f"\n  -> {verdict}")
