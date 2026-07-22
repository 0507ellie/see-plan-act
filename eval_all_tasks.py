import re
from pathlib import Path

import torch
import clip
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from train_bc import BCPolicy, build_lang_embed, rollout_episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/bc_best_seed42.pt"
NUM_EPISODES = 10
MAX_STEPS = 150
SUITE = "libero_object"
GIF_DIR = Path("eval_gifs_all_tasks")

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
policy = BCPolicy(proprio_dim=ckpt["proprio_dim"], chunk_size=ckpt["chunk_size"], hidden_dim=ckpt["hidden_dim"]).to(device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
action_mean = torch.tensor(ckpt["action_mean"], dtype=torch.float32).to(device)
action_std = torch.tensor(ckpt["action_std"], dtype=torch.float32).to(device)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
task_suite = benchmark.get_benchmark_dict()[SUITE]()

results = []
for task_id in range(task_suite.n_tasks):
    task = task_suite.get_task(task_id)
    lang_embed = build_lang_embed(clip_model, task.language, device)
    env = OffScreenRenderEnv(
        bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
        camera_heights=128, camera_widths=128,
        hard_reset=False,
    )
    slug = re.sub(r"\W+", "_", task.language).strip("_")
    task_dir = GIF_DIR / f"task{task_id}_{slug}"
    task_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for ep in range(NUM_EPISODES):
        success, frames = rollout_episode(
            policy, env, clip_model, clip_preprocess, lang_embed,
            action_mean, action_std, device, max_steps=MAX_STEPS, seq_len=10,
            temporal_ensemble=True, collect_frames=True,
        )
        successes += int(success)
        result = "success" if success else "fail"
        gif_path = task_dir / f"ep{ep}_{result}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    env.close()
    rate = successes / NUM_EPISODES
    results.append((task_id, task.language, successes, NUM_EPISODES, rate))
    print(f"task {task_id:2d} [{task.language}]: {successes}/{NUM_EPISODES} = {rate:.1%}", flush=True)

print("\n=== SUMMARY ===")
total_s = sum(r[2] for r in results)
total_n = sum(r[3] for r in results)
for task_id, lang, s, n, rate in results:
    print(f"  task {task_id}: {rate:.1%}  ({lang})")
print(f"Overall: {total_s}/{total_n} = {total_s/total_n:.1%}")
