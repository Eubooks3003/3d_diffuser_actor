"""Global-queue rollout eval across MULTIPLE multitask checkpoints and tasks.

Mirrors EC-Diffuser's eval_paper_mimicgen_multi_experiment.py: flattens every
(experiment x task) into ONE queue and keeps `--concurrency` workers busy across
GPUs, so a fast task freeing a slot immediately pulls the next queued job (even
across experiment boundaries). Each job = seeds x n_rollouts on fresh init
states. Resumable: a job whose output JSON already exists is skipped.

Runs on a sim-capable box (needs robosuite/mujoco). Example:
    python scripts/eval_queue.py \
        --ckpt_dir remote_ckpts --experiments baseline,single_action,uniform,random1 \
        --seeds 42,123,456 --n_rollouts 50 --gpus 0,1 --output_dir eval_results
"""
import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np

ALL12 = ["coffee_d0", "coffee_preparation_d0", "hammer_cleanup_d0", "kitchen_d0",
         "mug_cleanup_d0", "nut_assembly_d0", "pick_place_d0", "square_d0",
         "stack_d0", "stack_three_d0", "threading_d0", "three_piece_assembly_d0"]
REPO = str(Path(__file__).resolve().parent.parent)
ZOO = "/home/ellina/Desktop/robosuite-task-zoo"  # registers HammerCleanup/Kitchen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="remote_ckpts")
    p.add_argument("--experiments", default="baseline,single_action,uniform,random1")
    p.add_argument("--tasks", default=",".join(ALL12))
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--gpus", default="0")
    p.add_argument("--concurrency", type=int, default=0)  # 0 -> one per gpu
    p.add_argument("--output_dir", default="eval_results")
    args = p.parse_args()

    exps = [e for e in args.experiments.split(",") if e.strip()]
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    gpus = [g for g in args.gpus.split(",") if g.strip()]
    concurrency = args.concurrency or len(gpus)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Build the global queue of (experiment, task) jobs; skip already-done ones.
    queue = []
    for exp in exps:
        ckpt = Path(args.ckpt_dir) / f"multitask_tok_{exp}_best.pth"
        if not ckpt.exists():
            print(f"[skip] no checkpoint for '{exp}': {ckpt}", flush=True); continue
        for task in tasks:
            outp = out_dir / f"{exp}__{task}.json"
            if outp.exists():
                print(f"[done] {exp}/{task} (cached)", flush=True); continue
            queue.append({"exp": exp, "task": task, "ckpt": str(ckpt), "out": str(outp)})
    print(f"[queue] {len(queue)} jobs across {len(exps)} experiments x {len(tasks)} tasks; "
          f"concurrency={concurrency} gpus={gpus}", flush=True)

    def launch(job, gpu):
        pp = os.environ.get("PYTHONPATH", "")
        pp = f"{ZOO}:{pp}" if pp else ZOO  # task-zoo registers Hammer/Kitchen envs
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=pp, MUJOCO_GL="egl")
        cmd = ["xvfb-run", "-a", "python", "-u", "-m",
               "online_evaluation_mimicgen.eval_multitask_worker",
               "--checkpoint", job["ckpt"], "--experiment", job["exp"],
               "--task", job["task"], "--seeds", args.seeds,
               "--n_rollouts", str(args.n_rollouts), "--max_steps", str(args.max_steps),
               "--output", job["out"], "--device", "cuda"]
        log = open(out_dir / f"{job['exp']}__{job['task']}.log", "w")
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)

    # Pool: keep `concurrency` workers busy; round-robin GPU assignment.
    running, qi, gi = [], 0, 0
    while qi < len(queue) or running:
        while len(running) < concurrency and qi < len(queue):
            gpu = gpus[gi % len(gpus)]; gi += 1
            job = queue[qi]; qi += 1
            print(f"[launch] {job['exp']}/{job['task']} on gpu {gpu} "
                  f"({qi}/{len(queue)})", flush=True)
            running.append((launch(job, gpu), job))
        time.sleep(5)
        for proc, job in running[:]:
            if proc.poll() is not None:
                ok = proc.returncode == 0 and Path(job["out"]).exists()
                print(f"[{'ok' if ok else 'FAIL'}] {job['exp']}/{job['task']} "
                      f"(exit {proc.returncode})", flush=True)
                running.remove((proc, job))

    # Aggregate all JSONs -> per-experiment table.
    summary = {}
    for exp in exps:
        per_task, per_seed = {}, {}
        for task in tasks:
            fp = out_dir / f"{exp}__{task}.json"
            if not fp.exists():
                continue
            r = json.load(open(fp))
            per_task[task] = {"mean": r["mean"], "std": r["std"]}
            for ps in r["per_seed_results"]:
                per_seed.setdefault(int(ps["seed"]), []).append(float(ps["success_rate"]))
        across = {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_tasks": len(v)}
                  for s, v in per_seed.items()}
        overall = float(np.mean([t["mean"] for t in per_task.values()])) if per_task else 0.0
        summary[exp] = {"per_task": per_task, "across_task_per_seed": across,
                        "overall_mean_over_tasks": overall}

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUMMARY (mean success over tasks) ================")
    print(f"{'experiment':16s} {'overall':>8}   per-seed across-task mean")
    for exp, s in summary.items():
        seeds_str = "  ".join(f"s{sd}={d['mean']:.2f}" for sd, d in sorted(s["across_task_per_seed"].items()))
        print(f"{exp:16s} {s['overall_mean_over_tasks']:>8.3f}   {seeds_str}")
    print(f"\nfull table + per-task breakdown -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
