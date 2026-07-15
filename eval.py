import argparse
import importlib
import torch
import clip
from pathlib import Path
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POLICY_REGISTRY = {
    "bc": ("train_bc", "BCPolicy"),
}


def load_policy(policy_type, checkpoint_path):
    module_name, class_name = POLICY_REGISTRY[policy_type]
    module = importlib.import_module(module_name)
    policy_class = getattr(module, class_name)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = policy_class(
        proprio_dim=ckpt["proprio_dim"],
        chunk_size=ckpt.get("chunk_size", 8),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, ckpt, module


def evaluate(args):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    policy, ckpt, policy_module = load_policy(args.policy, args.checkpoint)
    action_mean = torch.tensor(ckpt["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(ckpt["action_std"], dtype=torch.float32).to(device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_id = args.task_id
    task = task_suite.get_task(task_id)

    lang_embed = policy_module.build_lang_embed(clip_model, task.language, device)

    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
        "camera_heights": 128,
        "camera_widths": 128,
    })

    gif_dir = Path(args.gif_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for ep in range(args.num_episodes):
        success, frames = policy_module.rollout_episode(
            policy, env, clip_model, clip_preprocess, lang_embed,
            action_mean, action_std, device,
            max_steps=args.max_steps, seq_len=args.seq_len,
            temporal_ensemble=args.temporal_ensemble, ensemble_decay=args.ensemble_decay,
            collect_frames=True,
        )
        successes += int(success)
        result = "success" if success else "fail"
        gif_path = gif_dir / f"ep{ep}_{result}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Episode {ep+1}/{args.num_episodes}: {result.upper()}  -> {gif_path}")

    success_rate = successes / args.num_episodes
    print(f"\nTask: {task.language}")
    print(f"Success rate: {successes}/{args.num_episodes} ({success_rate:.1%})")
    env.close()
    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, choices=list(POLICY_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seq_len", type=int, default=10, help="must match training seq_len")
    parser.add_argument("--temporal_ensemble", action="store_true", default=True)
    parser.add_argument("--no_temporal_ensemble", dest="temporal_ensemble", action="store_false")
    parser.add_argument("--ensemble_decay", type=float, default=0.1)
    parser.add_argument("--gif_dir", type=str, default="eval_gifs")
    args = parser.parse_args()
    evaluate(args)
