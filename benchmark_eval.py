"""Rigorous success-rate benchmarking, following LIBERO's own eval protocol
(libero/lifelong/evaluate.py): fixed init states per trial (not random env.reset()),
N trials/task cycling through the shipped init_files, per-task and suite-level
success rate with Clopper-Pearson 95% confidence intervals, full per-trial results
saved to JSON for reproducibility.

Usage:
    python benchmark_eval.py --policy bc --checkpoint checkpoints/bc_best_seed42.pt \
        --suite libero_object --num_trials 50
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import clip
import numpy as np
import torch
from scipy.stats import beta

from libero.libero import benchmark, get_libero_path

from eval import POLICY_REGISTRY, load_policy
from train_bc import build_lang_embed, rollout_episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clopper_pearson_ci(successes, n, alpha=0.05):
    """Exact binomial confidence interval. Returns (lo, hi) as fractions."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lo), float(hi)


def load_init_states(task):
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(path, weights_only=False)
    # Shuffled so that `init_states[trial % n_available]` with num_trials < n_available draws a
    # random subset instead of always the same fixed first-N states in file order. That fixed
    # slice turned out to be a real, reproducible source of bias: task 0 measured 20% (2/10) on
    # the unshuffled first-10 slice vs. 60% (30/50) on the full set, not sampling noise, the
    # same hard subset being reused identically on every eval call. A no-op when num_trials
    # equals the full 50 (order doesn't matter once every state is used).
    return init_states[np.random.permutation(len(init_states))]


def evaluate_task(policy, env, clip_model, clip_preprocess, lang_embed,
                   action_mean, action_std, init_states, num_trials,
                   max_steps, seq_len, temporal_ensemble, ensemble_decay):
    n_available = init_states.shape[0]
    outcomes = []
    for trial in range(num_trials):
        init_state = init_states[trial % n_available]
        success, _ = rollout_episode(
            policy, env, clip_model, clip_preprocess, lang_embed,
            action_mean, action_std, device,
            max_steps=max_steps, seq_len=seq_len,
            temporal_ensemble=temporal_ensemble, ensemble_decay=ensemble_decay,
            collect_frames=False, init_state=init_state,
        )
        outcomes.append(bool(success))
    return outcomes


def main(args):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    policy, ckpt, policy_module = load_policy(args.policy, args.checkpoint)
    action_mean = torch.tensor(ckpt["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(ckpt["action_std"], dtype=torch.float32).to(device)

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = (
        [int(t) for t in args.task_ids.split(",")]
        if args.task_ids
        else list(range(task_suite.n_tasks))
    )

    from libero.libero.envs import OffScreenRenderEnv

    per_task = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        init_states = load_init_states(task)
        n_available = init_states.shape[0]
        if args.num_trials > n_available:
            print(
                f"[warn] task {task_id}: requested {args.num_trials} trials but only "
                f"{n_available} init states shipped, cycling through them with repeats"
            )

        lang_embed = build_lang_embed(clip_model, task.language, device)
        env = OffScreenRenderEnv(
            bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
            camera_heights=args.camera_size, camera_widths=args.camera_size,
            hard_reset=False,
        )
        outcomes = evaluate_task(
            policy, env, clip_model, clip_preprocess, lang_embed,
            action_mean, action_std, init_states, args.num_trials,
            args.max_steps, args.seq_len, args.temporal_ensemble, args.ensemble_decay,
        )
        env.close()

        successes = sum(outcomes)
        n = len(outcomes)
        rate = successes / n
        lo, hi = clopper_pearson_ci(successes, n, args.alpha)
        per_task.append({
            "task_id": task_id,
            "language": task.language,
            "trials": n,
            "successes": successes,
            "success_rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "outcomes": outcomes,
        })
        print(
            f"task {task_id:2d} [{task.language[:50]:50s}] "
            f"{successes:3d}/{n:3d} = {rate:6.1%}  "
            f"95% CI [{lo:.1%}, {hi:.1%}]"
        )

    rates = np.array([t["success_rate"] for t in per_task])
    total_successes = sum(t["successes"] for t in per_task)
    total_trials = sum(t["trials"] for t in per_task)
    pooled_rate = total_successes / total_trials
    pooled_lo, pooled_hi = clopper_pearson_ci(total_successes, total_trials, args.alpha)

    print(f"\n=== SUMMARY ({args.suite}, checkpoint={args.checkpoint}) ===")
    print(f"Macro-avg success rate (mean of per-task rates): {rates.mean():.1%} (std {rates.std():.1%})")
    print(f"Pooled success rate: {total_successes}/{total_trials} = {pooled_rate:.1%}  "
          f"95% CI [{pooled_lo:.1%}, {pooled_hi:.1%}]")

    out_path = Path(args.out) if args.out else Path(args.out_dir) / (
        f"{args.suite}_{Path(args.checkpoint).stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "policy": args.policy,
            "suite": args.suite,
            "num_trials_requested": args.num_trials,
            "max_steps": args.max_steps,
            "seq_len": args.seq_len,
            "temporal_ensemble": args.temporal_ensemble,
            "ensemble_decay": args.ensemble_decay,
            "alpha": args.alpha,
            "per_task": per_task,
            "macro_avg_success_rate": float(rates.mean()),
            "macro_std_success_rate": float(rates.std()),
            "pooled_success_rate": pooled_rate,
            "pooled_ci_lo": pooled_lo,
            "pooled_ci_hi": pooled_hi,
        }, f, indent=2)
    print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, choices=list(POLICY_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task_ids", type=str, default="",
                         help="comma-separated task ids, default all tasks in suite")
    parser.add_argument("--num_trials", type=int, default=50,
                         help="trials per task; LIBERO ships 50 fixed init states per task")
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--seq_len", type=int, default=10, help="must match training seq_len")
    parser.add_argument("--camera_size", type=int, default=128)
    parser.add_argument("--temporal_ensemble", action="store_true", default=True)
    parser.add_argument("--no_temporal_ensemble", dest="temporal_ensemble", action="store_false")
    parser.add_argument("--ensemble_decay", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level for CIs")
    parser.add_argument("--out", type=str, default="", help="explicit output json path")
    parser.add_argument("--out_dir", type=str, default="benchmark_results")
    args = parser.parse_args()
    main(args)
