"""Residual SAC fine-tuning of a frozen BC checkpoint, modeled on the "PLD: Probe, Learn,
Distill" recipe (Lin et al., arxiv 2511.00091). Unlike train_ppo.py's on-policy PPO, this
trains one off-policy SAC specialist per task (see --task_ids), using:
  - A hybrid "probing" rollout: for a random prefix of each episode (T_base ~ Uniform(0,
    alpha_probe * max_raw_steps)), the frozen base acts alone. Only the remaining portion,
    where the residual specialist is active, becomes training data. This biases training
    toward states the base would actually visit in deployment.
  - A twin-Q critic + entropy-auto-tuned actor (standard SAC), operating on the same frozen
    base decoded features train_ppo.py already caches, using only chunk-step-0 of those
    features, since train_ppo.py found multi-step open-loop commitment collapses success
    on this BC checkpoint.
  - Reward shaping (touch/grasp/lift/release/no_touch/out_of_frame) reused verbatim from
    train_ppo.py's GraspAwareEnv.

Usage (single-task specialist):
    python train_sac.py --bc_checkpoint checkpoints/bc_best_seed42.pt \
        --suite libero_object --task_ids 0 --updates 200
"""
import argparse
import copy
import multiprocessing
import random
from datetime import datetime
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from train_bc import BCPolicy, build_lang_embed
from train_ppo import (
    GraspAwareEnv, encode_clip_images_batch, build_obs_features,
)
from benchmark_eval import load_init_states, evaluate_task
from sac_utils import ResidualSACActor, TwinQCritic, ReplayBuffer, soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualSACPolicy(nn.Module):
    """Structural analogue of train_ppo.py's FiLMResidualPolicy: wraps a frozen BCPolicy
    plus a ResidualSACActor. The TwinQCritic is not a submodule here since it has its own
    optimizer/target-network machinery with no place in an eval-time policy object.

    forward() keeps BCPolicy's exact (actions, done_logits) contract, so this plugs
    directly into rollout_episode/eval.py/benchmark_eval.py with no special-casing."""

    def __init__(self, proprio_dim=9, chunk_size=8, hidden_dim=256, action_dim=7,
                 lang_dim=512, film_hidden=256, xi=0.5,
                 init_log_std=-2.0, log_std_min=-5.0, log_std_max=2.0, **bc_kwargs):
        super().__init__()
        self.base = BCPolicy(
            proprio_dim=proprio_dim, chunk_size=chunk_size, hidden_dim=hidden_dim,
            action_dim=action_dim, **bc_kwargs,
        )
        for p in self.base.parameters():
            p.requires_grad = False
        self.chunk_size = chunk_size
        self.actor = ResidualSACActor(
            hidden_dim, lang_dim, action_dim, film_hidden, xi,
            init_log_std, log_std_min, log_std_max,
        )

    def freeze_base(self):
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # frozen base never leaves eval mode (BatchNorm/dropout)
        return self

    def _base_forward(self, clip_embed, images, proprio, lang):
        with torch.no_grad():
            base_actions, done_logits, decoded = self.base(
                clip_embed, images, proprio, lang, return_features=True,
            )
        return base_actions, done_logits, decoded

    def forward(self, clip_embed, images, proprio, lang):
        """Deterministic mean action, matches BCPolicy's eval-time contract. Only
        chunk-step-0 is corrected; see class docstring."""
        base_actions, done_logits, decoded = self._base_forward(clip_embed, images, proprio, lang)
        B, T = clip_embed.shape[0], clip_embed.shape[1]
        decoded_step0 = decoded.reshape(B, T, self.chunk_size, -1)[:, :, 0, :].reshape(B * T, -1)
        lang_exp = lang.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        det_residual = self.actor.act_deterministic(decoded_step0, lang_exp).reshape(B, T, -1)
        mean_actions = base_actions.clone()
        mean_actions[:, :, 0, :] = base_actions[:, :, 0, :] + det_residual
        return mean_actions, done_logits


def _query_base_step0(policy, clip_model, clip_preprocess, obs_batch,
                       clip_hist, img_hist, proprio_hist, lang_batch, seq_len, device):
    """Analogue of train_ppo.py's _query_policy_batch, trimmed to just what SAC needs:
    batched CLIP-encode + build_obs_features + history push/pad + frozen-base query.
    Returns (decoded_step0, base_action_0), each (M, hidden_dim)/(M, action_dim)."""
    M = len(obs_batch)
    ag_clip = encode_clip_images_batch(
        clip_model, clip_preprocess, [obs_batch[m]["agentview_image"] for m in range(M)], device,
    )
    eih_clip = encode_clip_images_batch(
        clip_model, clip_preprocess, [obs_batch[m]["robot0_eye_in_hand_image"] for m in range(M)], device,
    )
    clip_embed_now = torch.cat([ag_clip, eih_clip], dim=1)  # (M, 1024)

    images_now, proprio_now = [], []
    for m in range(M):
        imgs_m, proprio_m = build_obs_features(obs_batch[m], device)
        images_now.append(imgs_m)
        proprio_now.append(proprio_m)
    images_now = torch.stack(images_now)
    proprio_now = torch.stack(proprio_now)

    for m in range(M):
        clip_hist[m].append(clip_embed_now[m])
        img_hist[m].append(images_now[m])
        proprio_hist[m].append(proprio_now[m])
        clip_hist[m] = clip_hist[m][-seq_len:]
        img_hist[m] = img_hist[m][-seq_len:]
        proprio_hist[m] = proprio_hist[m][-seq_len:]

    pads = [seq_len - len(clip_hist[m]) for m in range(M)]
    clip_seq = torch.stack([torch.stack([clip_hist[m][0]] * pads[m] + clip_hist[m]) for m in range(M)])
    img_seq = torch.stack([torch.stack([img_hist[m][0]] * pads[m] + img_hist[m]) for m in range(M)])
    proprio_seq = torch.stack([torch.stack([proprio_hist[m][0]] * pads[m] + proprio_hist[m]) for m in range(M)])

    base_actions, _, decoded = policy._base_forward(clip_seq, img_seq, proprio_seq, lang_batch)
    base_action_0 = base_actions[:, -1, 0, :]                                  # (M, action_dim)
    decoded_last = decoded.reshape(M, seq_len, *decoded.shape[1:])[:, -1]      # (M, chunk_size, hidden_dim)
    decoded_step0 = decoded_last[:, 0, :]                                     # (M, hidden_dim)
    return decoded_step0, base_action_0


@torch.no_grad()
def collect_transitions_parallel(policy, vec_env, lang_embed, task_id, clip_model, clip_preprocess,
                                  action_mean, action_std, device, max_raw_steps, seq_len,
                                  alpha_probe, buffer, touch_bonus, grasp_bonus, lift_bonus,
                                  release_bonus, release_grace_steps, no_touch_penalty,
                                  out_of_frame_penalty):
    """Hybrid probing collector, PLD's core rollout recipe. All N parallel envs are bound
    to the same task_id (a SAC specialist trains on one task at a time). For each env,
    samples T_base ~ Uniform(0, alpha_probe * max_raw_steps) once per episode; while
    steps_raw < T_base, the frozen base acts alone (pure probing, not training data);
    once steps_raw >= T_base, a residual is sampled from policy.actor and the transition
    is pushed into `buffer`. alpha_probe=0.0 degenerates to "residual active from t=0".

    Reuses the same reward-shaping call sequence as train_ppo.py's
    collect_episodes_parallel (touch/grasp/lift/release/no_touch/out_of_frame).

    Only chunk-step-0 features/actions are queried (via _query_base_step0) and only that
    step is ever residual-corrected, matching ResidualSACActor's scope.

    Returns a stats dict: n_transitions actually pushed into `buffer`, n_episodes, n_success,
    mean_return (mirrors train_ppo.py's own success_rate/mean_return logging fields)."""
    N = len(vec_env)
    obs_list = vec_env.reset()

    clip_hist = [[] for _ in range(N)]
    img_hist = [[] for _ in range(N)]
    proprio_hist = [[] for _ in range(N)]
    touch_given = [False] * N
    grasp_given = [False] * N
    lift_given = [False] * N
    release_given = [False] * N
    out_of_frame_given = [False] * N
    episode_done = [False] * N
    raw_success = [False] * N
    steps_raw = [0] * N
    last_grasped_step = [-10**9] * N
    ep_return = [0.0] * N

    T_base = [random.uniform(0, alpha_probe * max_raw_steps) for _ in range(N)]
    lang_batch = lang_embed.expand(N, -1).to(device)

    n_transitions = 0

    decoded_step0, base_action_0 = _query_base_step0(
        policy, clip_model, clip_preprocess, obs_list, clip_hist, img_hist, proprio_hist,
        lang_batch, seq_len, device,
    )

    while not all(episode_done):
        probing = [steps_raw[i] < T_base[i] for i in range(N)]
        residual, _, _ = policy.actor.sample(decoded_step0, lang_batch)

        action_norm = base_action_0.clone()
        for i in range(N):
            if episode_done[i]:
                continue
            if not probing[i]:
                action_norm[i] = base_action_0[i] + residual[i]
            # else: probing[i] True -> action_norm[i] stays exactly base_action_0[i]

        actions_denorm = (action_norm * action_std + action_mean).cpu().numpy()
        # Defensive clip: the frozen base alone is trained to match valid demo actions and
        # never needs this, but a bounded-nonzero residual could in principle push the
        # composed action outside the env's accepted range.
        actions_denorm = np.clip(actions_denorm, -1.0, 1.0)
        for i in range(N):
            if episode_done[i]:
                actions_denorm[i] = 0.0

        obs_stack, rew_stack, done_stack, info_stack = vec_env.step(actions_denorm)

        # Query once on the new obs, reused both as this transition's s' AND as next
        # iteration's s, avoiding a redundant second frozen-base forward pass per raw step.
        next_decoded_step0, next_base_action_0 = _query_base_step0(
            policy, clip_model, clip_preprocess, obs_stack, clip_hist, img_hist, proprio_hist,
            lang_batch, seq_len, device,
        )

        for i in range(N):
            if episode_done[i]:
                continue
            steps_raw[i] += 1
            r = float(rew_stack[i])
            raw_success[i] = raw_success[i] or (r > 0)
            step_reward = r
            info_i = info_stack[i]

            if touch_bonus > 0 and not touch_given[i] and info_i.get("touched"):
                step_reward += touch_bonus
                touch_given[i] = True
            if info_i.get("grasped"):
                last_grasped_step[i] = steps_raw[i]
            if grasp_bonus > 0 and not grasp_given[i] and info_i.get("grasped"):
                step_reward += grasp_bonus
                grasp_given[i] = True
            if lift_bonus > 0 and not lift_given[i] and grasp_given[i] and info_i.get("lifted"):
                step_reward += lift_bonus
                lift_given[i] = True
            gripper_open_i = obs_stack[i]["robot0_gripper_qpos"][0] > 0.03
            recently_grasped = (steps_raw[i] - last_grasped_step[i]) <= release_grace_steps
            if (release_bonus > 0 and not release_given[i] and recently_grasped
                    and gripper_open_i and info_i.get("near_container")
                    and info_i.get("object_near_container")):
                step_reward += release_bonus
                release_given[i] = True
            if (out_of_frame_penalty > 0 and not out_of_frame_given[i]
                    and info_i.get("out_of_workspace")):
                step_reward -= out_of_frame_penalty
                out_of_frame_given[i] = True

            # terminal_i (LIBERO's own _check_success()-gated done_stack) is the ONLY correct
            # signal to zero out SAC's Bellman bootstrap on, a true absorbing state, exactly
            # like train_ppo.py's own bootstrap_value distinction. steps_raw[i] >= max_raw_steps
            # is an artificial wall-clock cutoff, not a real end of the MDP: the episode would
            # have continued accruing reward past this point if allowed to. Conflating the two
            # (as an earlier version of this function did, storing episode_done itself as the
            # buffer's `done`) would teach the critic that reward literally becomes unavailable
            # at max_raw_steps, biasing Q-estimates for any transition near the time limit
            # exactly the mistake "Time Limits in Reinforcement Learning" (Pardo et al. 2018)
            # documents, and exactly what train_ppo.py's own bootstrap_value machinery exists to
            # avoid on the PPO side.
            terminal_i = bool(done_stack[i])
            done_i = terminal_i or steps_raw[i] >= max_raw_steps
            if done_i:
                episode_done[i] = True
                if no_touch_penalty > 0 and not touch_given[i]:
                    step_reward -= no_touch_penalty

            ep_return[i] += step_reward

            if not probing[i]:
                buffer.add(
                    decoded_step0[i], next_decoded_step0[i], base_action_0[i],
                    next_base_action_0[i], action_norm[i], step_reward, terminal_i, task_id,
                )
                n_transitions += 1

        decoded_step0, base_action_0 = next_decoded_step0, next_base_action_0
        obs_list = obs_stack

    return {
        "n_transitions": n_transitions,
        "n_episodes": N,
        "n_success": sum(raw_success),
        "success_rate": sum(raw_success) / N,
        "mean_return": float(np.mean(ep_return)),
    }


def sac_update(policy, critic, target_critic, actor_optimizer, critic_optimizer,
               log_alpha, alpha_optimizer, target_entropy, batch, lang, gamma, tau,
               max_grad_norm=1.0):
    """One SAC gradient step (twin-Q critic + actor + auto-tuned entropy temperature) on one
    sampled minibatch. `batch` is a dict from ReplayBuffer.sample() (already on `device`);
    `lang` is resolved by the caller (constant per single-task specialist here) and passed
    in directly, since this function has no access to a {task_id: lang_embed} mapping.

    max_grad_norm=1.0 matches PLD's reported hyperparameter, applied to both the actor
    and critic optimizers. This is vanilla SAC, no CQL/Cal-QL conservative penalty.
    """
    decoded = batch["decoded"]
    next_decoded = batch["next_decoded"]
    base_action = batch["base_action"]
    next_base_action = batch["next_base_action"]
    action = batch["action"]
    reward = batch["reward"]
    done = batch["done"].float()

    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        next_residual, next_log_prob, _ = policy.actor.sample(next_decoded, lang)
        next_action = next_base_action + next_residual
        target_q1, target_q2 = target_critic(next_decoded, lang, next_action)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
        target_value = reward + (1.0 - done) * gamma * target_q

    q1, q2 = critic(decoded, lang, action)
    critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(q2, target_value)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_optimizer.step()

    residual, log_prob, _ = policy.actor.sample(decoded, lang)
    new_action = base_action + residual
    q1_new, q2_new = critic(decoded, lang, new_action)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha * log_prob - q_new).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(policy.actor.parameters(), max_grad_norm)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    soft_update(target_critic, critic, tau)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha": log_alpha.exp().item(),
        "alpha_loss": alpha_loss.item(),
        "mean_q": q_new.mean().item(),
        "mean_abs_residual": residual.abs().mean().item(),
    }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_name = args.run_name or datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Run name: {run_name}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    hidden_dim = bc_ckpt.get("hidden_dim", 256)
    chunk_size = bc_ckpt.get("chunk_size", 8)
    action_dim = bc_ckpt.get("action_dim", 7)
    lang_dim = bc_ckpt.get("lang_dim", 512)

    policy = ResidualSACPolicy(
        proprio_dim=bc_ckpt["proprio_dim"], chunk_size=chunk_size, hidden_dim=hidden_dim,
        action_dim=action_dim, lang_dim=lang_dim, film_hidden=args.film_hidden, xi=args.xi,
        init_log_std=args.init_log_std, log_std_min=args.log_std_min, log_std_max=args.log_std_max,
    ).to(device)
    policy.base.load_state_dict(bc_ckpt["policy_state_dict"])
    policy.freeze_base()

    action_mean = torch.tensor(bc_ckpt["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(bc_ckpt["action_std"], dtype=torch.float32).to(device)

    task_ids = [int(t) for t in args.task_ids.split(",")]
    assert len(task_ids) == 1, (
        "train_sac.py trains one SAC specialist per task (matching PLD's own per-task "
        "specialist structure), pass exactly one --task_ids value. To train specialists "
        "for every task in a suite, use run_specialists.py to loop this script over each "
        "task id sequentially."
    )
    task_id = task_ids[0]

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(task_id)
    lang_embed = build_lang_embed(clip_model, task.language, device)  # (1, lang_dim)
    print(f"Training SAC specialist for task {task_id}: {task.language}")

    def make_env_fn():
        return GraspAwareEnv(
            lift_height=args.lift_height,
            container_distance=args.basket_penalty_distance,
            workspace_margin_xy=args.workspace_margin_xy,
            bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
            camera_heights=args.camera_size, camera_widths=args.camera_size,
            hard_reset=False,
        )

    vec_env = SubprocVectorEnv([make_env_fn for _ in range(args.envs_per_task)])

    critic = TwinQCritic(hidden_dim, lang_dim, action_dim, film_hidden=args.film_hidden).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    for p in target_critic.parameters():
        p.requires_grad = False

    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = -args.target_entropy_scale * action_dim

    buffer = ReplayBuffer(args.buffer_capacity, hidden_dim, action_dim, device)

    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_eval_rate = -1.0

    try:
        for update in range(args.updates):
            policy.eval()
            stats = collect_transitions_parallel(
                policy, vec_env, lang_embed, task_id, clip_model, clip_preprocess,
                action_mean, action_std, device, args.max_raw_steps, args.seq_len,
                args.alpha_probe, buffer,
                touch_bonus=args.touch_bonus, grasp_bonus=args.grasp_bonus, lift_bonus=args.lift_bonus,
                release_bonus=args.release_bonus, release_grace_steps=args.release_grace_steps,
                no_touch_penalty=args.no_touch_penalty, out_of_frame_penalty=args.out_of_frame_penalty,
            )

            policy.train()
            update_stats = None
            if len(buffer) >= args.batch_size:
                for _ in range(args.updates_per_round):
                    batch = buffer.sample(args.batch_size)
                    lang_for_batch = lang_embed.expand(args.batch_size, -1).to(device)
                    update_stats = sac_update(
                        policy, critic, target_critic, actor_optimizer, critic_optimizer,
                        log_alpha, alpha_optimizer, target_entropy, batch, lang_for_batch,
                        args.gamma, args.tau, max_grad_norm=args.max_grad_norm,
                    )

            log_line = (
                f"[update {update+1}/{args.updates}] rollout success={stats['success_rate']:.1%} "
                f"return={stats['mean_return']:.3f} buffer={len(buffer)}"
            )
            if update_stats is not None:
                log_line += (
                    f" critic_loss={update_stats['critic_loss']:.4f} "
                    f"actor_loss={update_stats['actor_loss']:.4f} alpha={update_stats['alpha']:.4f} "
                    f"mean_q={update_stats['mean_q']:.4f} mean|residual|={update_stats['mean_abs_residual']:.4f}"
                )
            print(log_line)

            do_eval = args.eval_every > 0 and ((update + 1) % args.eval_every == 0 or update == args.updates - 1)
            eval_rate = None
            if do_eval:
                eval_env = OffScreenRenderEnv(
                    bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
                    camera_heights=args.camera_size, camera_widths=args.camera_size,
                    hard_reset=False,
                )
                try:
                    init_states = load_init_states(task)
                    policy.eval()
                    outcomes = evaluate_task(
                        policy, eval_env, clip_model, clip_preprocess, lang_embed,
                        action_mean, action_std, init_states, args.eval_trials,
                        args.max_raw_steps, args.seq_len, temporal_ensemble=True, ensemble_decay=0.1,
                    )
                finally:
                    eval_env.close()
                eval_rate = sum(outcomes) / len(outcomes)
                print(f"  [eval @ update {update+1}] task {task_id} success={eval_rate:.1%} "
                      f"({sum(outcomes)}/{len(outcomes)})")

            do_save = (args.save_every > 0 and (update + 1) % args.save_every == 0) or update == args.updates - 1
            is_new_best = eval_rate is not None and eval_rate > best_eval_rate
            if do_save or is_new_best:
                ckpt = {
                    "policy_state_dict": policy.state_dict(),
                    "action_mean": bc_ckpt["action_mean"],
                    "action_std": bc_ckpt["action_std"],
                    "proprio_dim": bc_ckpt["proprio_dim"],
                    "chunk_size": chunk_size,
                    "hidden_dim": hidden_dim,
                    "action_dim": action_dim,
                    "lang_dim": lang_dim,
                    "film_hidden": args.film_hidden,
                    "xi": args.xi,
                    "log_std_min": args.log_std_min,
                    "log_std_max": args.log_std_max,
                    "suite": args.suite,
                    "task_ids": task_ids,
                    "bc_checkpoint": args.bc_checkpoint,
                    "eval_success_rate": eval_rate,
                }
                if do_save:
                    ckpt_path = save_dir / f"residual_sac_seed{args.seed}_update{update+1}.pt"
                    torch.save(ckpt, ckpt_path)
                    print(f"  saved {ckpt_path}")
                if is_new_best:
                    best_eval_rate = eval_rate
                    best_path = save_dir / f"residual_sac_seed{args.seed}_best.pt"
                    torch.save(ckpt, best_path)
                    print(f"  new best (eval success={eval_rate:.1%}) -> {best_path}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    # Same CUDA-fork-safety reasoning as train_ppo.py's __main__ block: forkserver must be
    # set up before any CUDA init (clip.load below) and before SubprocVectorEnv forks
    # workers, since forking a process that already holds a CUDA context is a well-known
    # source of silent hangs/crashes in the child.
    multiprocessing.set_forkserver_preload(["libero.libero.envs", "robosuite"])
    multiprocessing.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task_ids", type=str, required=True,
                         help="exactly one task id, PLD trains SAC specialists strictly "
                              "per-task, one at a time (see run_specialists.py to loop this "
                              "over every task in a suite).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--camera_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=10, help="must match BC training seq_len")
    parser.add_argument("--max_raw_steps", type=int, default=300)

    # Reward shaping, same flag names/defaults as train_ppo.py, so a user tuning both
    # scripts doesn't need two different flag vocabularies for one shaping scheme.
    parser.add_argument("--touch_bonus", type=float, default=0.12)
    parser.add_argument("--grasp_bonus", type=float, default=0.15)
    parser.add_argument("--lift_bonus", type=float, default=0.2)
    parser.add_argument("--lift_height", type=float, default=0.04)
    parser.add_argument("--release_bonus", type=float, default=0.25)
    parser.add_argument("--release_grace_steps", type=int, default=10)
    parser.add_argument("--basket_penalty_distance", type=float, default=0.15)
    parser.add_argument("--no_touch_penalty", type=float, default=0.3)
    parser.add_argument("--out_of_frame_penalty", type=float, default=0.15)
    parser.add_argument("--workspace_margin_xy", type=float, default=0.4)

    parser.add_argument("--alpha_probe", type=float, default=0.6,
                         help="PLD's hybrid-probing alpha: T_base ~ Uniform(0, alpha_probe * "
                              "max_raw_steps) raw steps of pure frozen-base rollout before "
                              "the residual specialist takes over each episode. 0 disables "
                              "probing entirely.")
    parser.add_argument("--xi", type=float, default=0.5,
                         help="hard bound on |residual| in normalized action space "
                              "(PLD's default of 0.5 for LIBERO).")
    parser.add_argument("--film_hidden", type=int, default=256,
                         help="width of the actor/critic's FiLM-conditioned trunk, matching "
                              "PLD's own reported hidden-dim (paper Table 5) for their "
                              "lightweight residual actor/critic network, note this is a "
                              "different, smaller network than the frozen base's own "
                              "hidden_dim, same relationship train_ppo.py's "
                              "FiLMResidualHead film_hidden has to the base.")
    parser.add_argument("--init_log_std", type=float, default=-2.0)
    parser.add_argument("--log_std_min", type=float, default=-5.0)
    parser.add_argument("--log_std_max", type=float, default=2.0)
    parser.add_argument("--target_entropy_scale", type=float, default=0.5,
                         help="target_entropy = -target_entropy_scale * action_dim, PLD default 0.5")

    parser.add_argument("--envs_per_task", type=int, default=4,
                         help="parallel subprocess workers collecting transitions for this "
                              "one task's specialist.")
    parser.add_argument("--buffer_capacity", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak target-network update rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                         help="gradient clipping norm for both the actor and critic optimizers, "
                              "matching PLD's own reported hyperparameter (paper Table 5).")
    parser.add_argument("--updates", type=int, default=500,
                         help="rollout-collection rounds; each round collects one episode/"
                              "env then runs --updates_per_round SAC gradient steps")
    parser.add_argument("--updates_per_round", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)

    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_trials", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints_sac")
    parser.add_argument("--save_every", type=int, default=50)

    args = parser.parse_args()
    train(args)
