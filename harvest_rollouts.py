"""Harvest rollout trajectories from a trained SAC specialist checkpoint, for later LoRA
distillation (see distill_lora.py). Inference-only: runs the already-trained policy for
--num_episodes on its own task, recording the pre-adapter features (resnet_out, clip_embed,
before their respective LoRA adapters) that distillation needs as its retraining input,
since train_sac.py's cached post-decoder `decoded` depends on the old adapter weights and
would be invalidated the moment distillation changes them.

Each harvested episode is windowed the same way train_bc.py's DemoDataset windows a human
demo (act[i:i+chunk_size], dones[i:i+chunk_size].max()), by distill_lora.py's RolloutDataset.

Temporal ensembling defaults to ON, matching rollout_episode's own default. An earlier
version of this file disabled it to match training's own convention, which was wrong:
training explores via stochastic sampling, and it was that noise, not the absence of
ensembling, that made non-ensembled rollouts work during training. With the deterministic
mean action and ensembling off, one specialist (task 1) collapsed from a verified 90% eval
rate to 0/50; re-enabling ensembling (the same algorithm rollout_episode uses, done in
normalized space here) fixed it.

Usage:
    python harvest_rollouts.py --bc_checkpoint checkpoints/bc_best_seed42.pt \
        --sac_checkpoint checkpoints_sac/task0/residual_sac_seed42_best.pt \
        --task_ids 0 --num_episodes 50 --out_dir rollout_shards
"""
import argparse
import multiprocessing
from pathlib import Path

import clip
import numpy as np
import torch

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from train_bc import build_lang_embed, RESNET_PREPROCESS
from train_ppo import build_obs_features, encode_clip_images_batch
from train_sac import ResidualSACPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _query_step(policy, clip_model, clip_preprocess, obs, clip_hist, img_hist, proprio_hist,
                 lang, seq_len, device):
    """Single-env analogue of train_sac.py's _query_base_step0, extended to also return
    the pre-adapter resnet_out/clip_embed for this step, which is what distillation needs
    cached (BCPolicy.forward() never exposes these directly).

    Mutates clip_hist/img_hist/proprio_hist in place. Returns (mean_actions_chunk,
    resnet_out_now, clip_embed_now, proprio_now), where mean_actions_chunk is the full
    (chunk_size, action_dim) predicted chunk (not just chunk-step-0), since temporal
    ensembling needs every step to blend across overlapping queries."""
    clip_embed_now = encode_clip_images_batch(
        clip_model, clip_preprocess, [obs["agentview_image"], obs["robot0_eye_in_hand_image"]], device,
    )  # (2, 512), one row per view; NOT yet the (1,1024) concatenated form _query_base_step0 uses
    clip_embed_flat = clip_embed_now.reshape(1, -1)  # (1, 1024), matches downstream convention

    images_now, proprio_now = build_obs_features(obs, device)  # images_now: (2,3,224,224)
    resnet_out_now = policy.base.resnet(images_now)  # (2, 512), PRE-adapter, resnet_adapter not applied

    clip_hist.append(clip_embed_flat[0])
    img_hist.append(images_now)
    proprio_hist.append(proprio_now)
    clip_hist[:] = clip_hist[-seq_len:]
    img_hist[:] = img_hist[-seq_len:]
    proprio_hist[:] = proprio_hist[-seq_len:]

    pad = seq_len - len(clip_hist)
    clip_seq = torch.stack([clip_hist[0]] * pad + clip_hist).unsqueeze(0)      # (1, seq_len, 1024)
    img_seq = torch.stack([img_hist[0]] * pad + img_hist).unsqueeze(0)          # (1, seq_len, 2, 3, 224, 224)
    proprio_seq = torch.stack([proprio_hist[0]] * pad + proprio_hist).unsqueeze(0)  # (1, seq_len, proprio_dim)
    lang_batch = lang.unsqueeze(0)  # (1, lang_dim)

    mean_actions, _ = policy.forward(clip_seq, img_seq, proprio_seq, lang_batch)  # (1,seq_len,chunk_size,action_dim)
    # -1, not 0: only the prediction tied to the most recent observation in the history
    # window is the one to act on (same convention as _query_base_step0).
    mean_actions_chunk = mean_actions[0, -1]  # (chunk_size, action_dim)

    return mean_actions_chunk, resnet_out_now, clip_embed_flat[0], proprio_now


@torch.no_grad()
def harvest_task(policy, env, clip_model, clip_preprocess, lang_embed, action_mean, action_std,
                  device, num_episodes, max_raw_steps, seq_len, task_id,
                  temporal_ensemble=True, ensemble_decay=0.1):
    """Runs num_episodes closed-loop rollouts of `policy` on `env`, recording per-raw-step
    (resnet_out, clip_embed, proprio, action, done) streams. Returns a list of per-episode
    dicts holding full-length numpy arrays (T, ...); windowing happens later in
    distill_lora.py's RolloutDataset.

    temporal_ensemble=True (default) blends the current query's chunk-step-0 prediction
    with overlapping later-chunk-step predictions from recent past queries, same algorithm
    as rollout_episode (see module docstring for why this defaults on)."""
    episodes = []
    n_success = 0
    chunk_size = policy.chunk_size
    action_mean_np = action_mean.cpu().numpy()
    action_std_np = action_std.cpu().numpy()
    for ep in range(num_episodes):
        obs = env.reset()
        clip_hist, img_hist, proprio_hist = [], [], []
        resnet_stream, clip_stream, proprio_stream, action_stream, done_stream = [], [], [], [], []
        chunk_history = {}
        success = False
        for t in range(max_raw_steps):
            mean_actions_chunk, resnet_out_now, clip_embed_now, proprio_now = _query_step(
                policy, clip_model, clip_preprocess, obs, clip_hist, img_hist, proprio_hist,
                lang_embed, seq_len, device,
            )
            chunk_np = mean_actions_chunk.cpu().numpy()  # (chunk_size, action_dim), normalized space

            if temporal_ensemble:
                # Same algorithm as rollout_episode (train_bc.py), kept in normalized space
                # (equivalent either way, see module docstring) so the recorded action stays
                # in action_head's own normalized output space.
                chunk_history[t] = chunk_np
                oldest = max(0, t - chunk_size + 1)
                for s in list(chunk_history):
                    if s < oldest:
                        del chunk_history[s]
                candidates, weights = [], []
                for s, chunk in chunk_history.items():
                    offset = t - s
                    if offset < len(chunk):
                        candidates.append(chunk[offset])
                        weights.append(np.exp(-ensemble_decay * offset))
                weights = np.array(weights) / np.sum(weights)
                action_norm = np.sum(np.stack(candidates) * weights[:, None], axis=0)
            else:
                action_norm = chunk_np[0]

            action_denorm = action_norm * action_std_np + action_mean_np
            action_denorm = np.clip(action_denorm, -1.0, 1.0)

            resnet_stream.append(resnet_out_now.cpu().numpy())
            clip_stream.append(clip_embed_now.cpu().numpy())
            proprio_stream.append(proprio_now.cpu().numpy())
            action_stream.append(action_norm)  # normalized space, the (possibly ensembled) action actually executed

            obs, r, done, info = env.step(action_denorm)
            done_stream.append(bool(done))
            if r > 0:
                success = True
            if done:
                break

        n_success += int(success)
        episodes.append({
            "resnet_out": np.array(resnet_stream, dtype=np.float32),   # (T, 2, 512)
            "clip_embed": np.array(clip_stream, dtype=np.float32),     # (T, 1024)
            "proprio": np.array(proprio_stream, dtype=np.float32),     # (T, proprio_dim)
            "action": np.array(action_stream, dtype=np.float32),       # (T, action_dim) normalized
            "done": np.array(done_stream, dtype=np.float32),           # (T,)
            "task_id": task_id,
            "success": success,
        })
        print(f"  episode {ep+1}/{num_episodes}: {'success' if success else 'fail'} "
              f"({len(done_stream)} steps)")

    print(f"Harvested {num_episodes} episodes for task {task_id}: "
          f"{n_success}/{num_episodes} succeeded")
    return episodes


def main(args):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    sac_ckpt = torch.load(args.sac_checkpoint, map_location=device, weights_only=False)

    hidden_dim = sac_ckpt.get("hidden_dim", bc_ckpt.get("hidden_dim", 256))
    chunk_size = sac_ckpt.get("chunk_size", bc_ckpt.get("chunk_size", 8))
    action_dim = sac_ckpt.get("action_dim", 7)
    lang_dim = sac_ckpt.get("lang_dim", 512)
    film_hidden = sac_ckpt.get("film_hidden", 256)
    xi = sac_ckpt.get("xi", 0.5)
    log_std_min = sac_ckpt.get("log_std_min", -5.0)
    log_std_max = sac_ckpt.get("log_std_max", 2.0)

    policy = ResidualSACPolicy(
        proprio_dim=bc_ckpt["proprio_dim"], chunk_size=chunk_size, hidden_dim=hidden_dim,
        action_dim=action_dim, lang_dim=lang_dim, film_hidden=film_hidden, xi=xi,
        log_std_min=log_std_min, log_std_max=log_std_max,
    ).to(device)
    policy.load_state_dict(sac_ckpt["policy_state_dict"])
    policy.eval()

    action_mean = torch.tensor(bc_ckpt["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(bc_ckpt["action_std"], dtype=torch.float32).to(device)

    task_ids = [int(t) for t in args.task_ids.split(",")]
    assert len(task_ids) == 1, "harvest one task at a time (matches one specialist checkpoint each)"
    task_id = task_ids[0]

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(task_id)
    lang_embed = build_lang_embed(clip_model, task.language, device).squeeze(0)  # (lang_dim,)
    print(f"Harvesting task {task_id}: {task.language}")

    env = OffScreenRenderEnv(
        bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
        camera_heights=args.camera_size, camera_widths=args.camera_size,
        hard_reset=False,
    )
    try:
        episodes = harvest_task(
            policy, env, clip_model, clip_preprocess, lang_embed, action_mean, action_std,
            device, args.num_episodes, args.max_raw_steps, args.seq_len, task_id,
            temporal_ensemble=args.temporal_ensemble, ensemble_decay=args.ensemble_decay,
        )
    finally:
        env.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"task{task_id}_shard.pt"
    torch.save({"episodes": episodes, "task_id": task_id, "language": task.language}, out_path)
    print(f"Saved {len(episodes)} episodes -> {out_path}")


if __name__ == "__main__":
    multiprocessing.set_forkserver_preload(["libero.libero.envs", "robosuite"])
    multiprocessing.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--sac_checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task_ids", type=str, required=True, help="exactly one task id")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_raw_steps", type=int, default=300)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--camera_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="rollout_shards")
    parser.add_argument("--temporal_ensemble", action="store_true", default=True,
                         help="matches rollout_episode's own default; see module docstring.")
    parser.add_argument("--no_temporal_ensemble", dest="temporal_ensemble", action="store_false")
    parser.add_argument("--ensemble_decay", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
