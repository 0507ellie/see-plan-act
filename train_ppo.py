"""Residual PPO fine-tuning of a frozen BC checkpoint, with FiLM (language-conditioned
feature-wise modulation) inside the residual head.

The pretrained BCPolicy is a deterministic action-chunking regressor with no value
head, so it's frozen entirely and a small residual module is trained on top instead:
base_action + residual(features, language). This is standard residual policy learning
(Silver et al. 2018; Johannink et al. 2018): the frozen controller already solves most
of the task, so PPO only has to learn a small, regularized correction.

Key design points:
  - The residual head's last layer is zero-initialized, so at the start of training
    the policy is bit-for-bit identical to the frozen BC policy.
  - FiLM lives inside the residual head: the language embedding produces a per-channel
    (gamma, beta) that modulates the pretrained policy's fused features before the
    residual/value heads consume them.
  - Each PPO decision commits to --replan_horizon raw env steps open-loop before the
    policy is re-queried (default 1, pure closed-loop). An empirical sweep on this BC
    checkpoint found success collapsing almost immediately past horizon 1, so it's
    kept closed-loop by default with the mechanism left configurable.
  - Since the frozen base is deterministic given an observation window, rollout
    collection caches its compact intermediate features once, and the PPO update
    runs off that cache instead of replaying raw images through CLIP/ResNet/LSTM
    every epoch.
  - A short critic-only warmup fits the value function before the actor starts
    updating, so early advantages aren't computed against a garbage value estimate.
    The actor update is also gated on the rollout batch containing at least one real
    success, since PPO's advantage normalization blows up floating-point noise into
    fake large advantages when every return in a batch is exactly 0.

Usage:
    python train_ppo.py --bc_checkpoint checkpoints/bc_best_seed42.pt \
        --suite libero_object --task_ids 0 --updates 200
"""
import argparse
import multiprocessing
import random
from datetime import datetime
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from robosuite.utils.transform_utils import quat2axisangle

from train_bc import BCPolicy, RESNET_PREPROCESS, build_lang_embed, rollout_episode
from benchmark_eval import load_init_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FiLMResidualHead(nn.Module):
    def __init__(self, hidden_dim, lang_dim, chunk_size, action_dim,
                 film_hidden=128, init_log_std=-2.0, log_std_min=-3.0, log_std_max=-1.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.feature_proj = nn.Linear(hidden_dim, film_hidden)
        self.film_generator = nn.Linear(lang_dim, 2 * film_hidden)
        self.trunk = nn.Sequential(nn.Linear(film_hidden, film_hidden), nn.ReLU())

        self.action_residual = nn.Linear(film_hidden, action_dim)
        nn.init.zeros_(self.action_residual.weight)
        nn.init.zeros_(self.action_residual.bias)

        self.value_head = nn.Linear(film_hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, decoded, lang_exp):
        # decoded: (N, chunk_size, hidden_dim), lang_exp: (N, lang_dim)
        h = self.feature_proj(decoded)
        gamma, beta = self.film_generator(lang_exp).chunk(2, dim=-1)
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)
        h = self.trunk(torch.relu(h))

        residual = self.action_residual(h)                    # (N, chunk_size, action_dim)
        value = self.value_head(h.mean(dim=1)).squeeze(-1)     # (N,)
        return residual, value

    def std(self):
        # Bounded so the entropy bonus can't inflate exploration noise without limit
        # when a run never observes a positive reward to trade off against it.
        return torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))


class FiLMResidualPolicy(nn.Module):
    """Drop-in replacement for BCPolicy: forward() has the exact same signature and
    deterministic (mean action, done_logits) contract, so it plugs straight into
    rollout_episode/quick_eval/benchmark_eval.py. evaluate_actions_from_cache() is
    the PPO-only entry point, driven entirely by cached features so the PPO update
    never re-runs the frozen base."""

    def __init__(self, proprio_dim=9, chunk_size=8, hidden_dim=256, action_dim=7,
                 lang_dim=512, film_hidden=128, init_log_std=-2.0,
                 log_std_min=-3.0, log_std_max=-1.0, **bc_kwargs):
        super().__init__()
        self.base = BCPolicy(
            proprio_dim=proprio_dim, chunk_size=chunk_size, hidden_dim=hidden_dim,
            action_dim=action_dim, **bc_kwargs,
        )
        for p in self.base.parameters():
            p.requires_grad = False
        self.chunk_size = chunk_size
        self.residual_head = FiLMResidualHead(
            hidden_dim, lang_dim, chunk_size, action_dim, film_hidden, init_log_std,
            log_std_min, log_std_max,
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

    def _residual(self, decoded, base_actions, B, T, lang):
        lang_exp = lang.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        residual, value = self.residual_head(decoded, lang_exp)
        residual = residual.reshape(B, T, self.chunk_size, -1)
        mean_actions = base_actions + residual
        value = value.reshape(B, T)
        return mean_actions, value

    def forward(self, clip_embed, images, proprio, lang):
        """Deterministic mean action, matches BCPolicy's eval-time contract."""
        base_actions, done_logits, decoded = self._base_forward(clip_embed, images, proprio, lang)
        B, T = clip_embed.shape[0], clip_embed.shape[1]
        mean_actions, _ = self._residual(decoded, base_actions, B, T, lang)
        return mean_actions, done_logits

    def evaluate_actions_from_cache(self, decoded_last_batch, base_actions_k_batch, lang_batch,
                                     actions_batch, mask_batch, values_only=False):
        """Recompute log_prob/entropy/value for stored macro-actions during a PPO
        update, entirely from cached _query_policy_batch() features (no frozen-base
        forward pass). actions_batch/mask_batch: (N, replan_horizon, action_dim) /
        (N, replan_horizon); padded steps near episode end don't contribute to the sums.

        Also returns residual_l2 (N,), the masked sum of squared residual magnitude,
        which lets the caller regularize the residual toward zero directly. This is
        separate from target_kl: target_kl only bounds movement per update, so many
        small approved steps can still drift the residual far from the base over many
        updates without residual_l2 anchoring it back.

        values_only=True skips the log_prob/entropy/residual_l2 computation once
        `value` (all critic_warmup_update needs) has been computed."""
        residual, value = self.residual_head(decoded_last_batch, lang_batch)  # (N,chunk_size,A), (N,)
        if values_only:
            return None, None, value, None
        rh = base_actions_k_batch.shape[1]
        residual_used = residual[:, :rh]                            # (N, replan_horizon, A)
        mean_k = base_actions_k_batch + residual_used
        dist = torch.distributions.Normal(mean_k, self.residual_head.std())
        log_prob_step = dist.log_prob(actions_batch).sum(-1)        # (N, replan_horizon)
        entropy_step = dist.entropy().sum(-1)                        # (N, replan_horizon)
        log_prob = (log_prob_step * mask_batch).sum(-1)              # (N,)
        entropy = (entropy_step * mask_batch).sum(-1)                # (N,)
        # Mean, not sum, over the valid (real, non-padded) scalar action entries, a raw sum's
        # magnitude scales with replan_horizon * action_dim, so residual_l2_coef's effective
        # strength would silently change if --replan_horizon were ever set above the default 1,
        # even though nothing about the intended "how far is the mean residual from zero, on
        # average per action component" penalty should depend on that.
        valid_dims = (mask_batch.sum(-1) * residual_used.shape[-1]).clamp(min=1)
        residual_l2 = (residual_used.pow(2).sum(-1) * mask_batch).sum(-1) / valid_dims  # (N,)
        return log_prob, entropy, value, residual_l2


def build_obs_features(obs, device):
    gripper_open = np.array([1.0 if obs["robot0_gripper_qpos"][0] > 0.03 else 0.0])
    proprio = np.concatenate([
        obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"], gripper_open,
    ])
    ag_img = RESNET_PREPROCESS(Image.fromarray(obs["agentview_image"]))
    eih_img = RESNET_PREPROCESS(Image.fromarray(obs["robot0_eye_in_hand_image"]))
    images = torch.stack([ag_img, eih_img]).to(device)
    return images, torch.tensor(proprio, dtype=torch.float32).to(device)


def find_container_name(target_names):
    """Heuristic for the destination container among a task's obj_of_interest:
    prefer a name containing "basket" (matches every libero_object task), else
    fall back to the last entry (the target-to-pick is conventionally listed
    before the destination in these bddl files)."""
    for name in target_names:
        if "basket" in name.lower():
            return name
    return target_names[-1] if target_names else None


def find_pickable_names(target_names, container_name):
    """obj_of_interest minus the destination container, the actual thing to
    grasp/touch, so touch/grasp/lift bonuses can't fire from contact with the
    basket itself."""
    return [n for n in target_names if n != container_name]


def target_touched(env, pickable_names):
    """True if any part of the gripper touches a pickable target. Weaker than
    target_grasped (which requires the two-fingerpad pinch pattern); catches the
    first contact attempt, a sub-goal earlier than a successful grasp."""
    sim_env = env.env
    gripper = sim_env.robots[0].gripper
    for name in pickable_names:
        obj = sim_env.objects_dict.get(name)
        if obj is not None and sim_env.check_contact(gripper, obj):
            return True
    return False


def target_grasped(env, pickable_names):
    """True if the gripper is grasping any pickable target (robosuite's own
    contact-based grasp check)."""
    sim_env = env.env
    gripper = sim_env.robots[0].gripper
    for name in pickable_names:
        obj = sim_env.objects_dict.get(name)
        if obj is not None and sim_env._check_grasp(gripper=gripper, object_geoms=obj):
            return True
    return False


def target_body_heights(env, target_names):
    """z-height of each target object's body, keyed by name. Used to detect a real
    lift (object raised off the table), not just reach/hover proximity."""
    sim_env = env.env
    heights = {}
    for name in target_names:
        body_id = sim_env.obj_body_id.get(name)
        if body_id is not None:
            heights[name] = float(sim_env.sim.data.body_xpos[body_id][2])
    return heights


def target_lifted(env, pickable_names, initial_heights, lift_height):
    current = target_body_heights(env, pickable_names)
    return any(
        name in initial_heights and current[name] - initial_heights[name] > lift_height
        for name in current
    )


def gripper_near_container(env, container_name, distance_threshold):
    """True if the gripper is within distance_threshold of the destination
    container. Distance-based, not contact-based, so hovering above the basket
    without touching it still counts. Paired at the caller with "hasn't grasped
    yet" so the real carry-to-basket approach is never penalized."""
    sim_env = env.env
    gripper = sim_env.robots[0].gripper
    container = sim_env.objects_dict.get(container_name) or sim_env.fixtures_dict.get(container_name)
    if container is None:
        return False
    dist = sim_env._gripper_to_target(
        gripper=gripper, target=container.root_body, target_type="body", return_distance=True,
    )
    return dist < distance_threshold


def object_near_container(env, pickable_names, container_name, distance_threshold):
    """True if any pickable target's own body (not the gripper) is within
    distance_threshold of the container. Needed because gating release on
    recent-grasp + gripper-near-container alone still lets an episode grasp,
    drop the object elsewhere, then rush an empty gripper to the basket and
    open, without the object ever coming near it."""
    sim_env = env.env
    container = sim_env.objects_dict.get(container_name) or sim_env.fixtures_dict.get(container_name)
    if container is None:
        return False
    container_pos = np.array(sim_env.sim.data.get_body_xpos(container.root_body), dtype=np.float64)
    for name in pickable_names:
        body_id = sim_env.obj_body_id.get(name)
        if body_id is None:
            continue
        obj_pos = np.array(sim_env.sim.data.body_xpos[body_id], dtype=np.float64)
        if np.linalg.norm(obj_pos - container_pos) < distance_threshold:
            return True
    return False


def get_workspace_bounds(env, margin_xy=0.4, margin_z_below=0.05, margin_z_above=0.6,
                          fallback_reach_xy=1.0, fallback_z_below=0.1, fallback_z_above=1.0):
    """A generous XYZ bounding box around the workspace, used to detect the
    end-effector straying well outside it (the "arm swings out and disengages"
    failure mode). Derived from the arena's own table dimensions when available.
    libero_object's EmptyArena exposes no table_offset/table_half_size, so it
    falls back to a bound centered on the robot's base position with a fixed
    reach-based half-extent."""
    arena = env.env.model.mujoco_arena
    if hasattr(arena, "table_offset") and hasattr(arena, "table_half_size"):
        offset = np.array(arena.table_offset, dtype=np.float64)
        half = np.array(arena.table_half_size, dtype=np.float64)
        lo = offset - half - np.array([margin_xy, margin_xy, margin_z_below])
        hi = offset + half + np.array([margin_xy, margin_xy, margin_z_above])
    else:
        base_pos = np.array(env.env.robots[0].base_pos, dtype=np.float64)
        half = np.array([fallback_reach_xy, fallback_reach_xy, fallback_reach_xy])
        lo = base_pos - half - np.array([0.0, 0.0, fallback_z_below])
        hi = base_pos + half + np.array([0.0, 0.0, fallback_z_above])
    return lo, hi


def gripper_out_of_workspace(obs, bounds):
    lo, hi = bounds
    pos = obs["robot0_eef_pos"]
    return bool(np.any(pos < lo) or np.any(pos > hi))


class GraspAwareEnv:
    """Wraps OffScreenRenderEnv so touch/grasp/lift/near-container status rides
    back through step()'s own info dict, computed locally wherever the env
    actually lives.

    SubprocVectorEnv runs each env in its own subprocess and only exposes a
    fixed RPC protocol (step/reset/getattr/setattr/close/...), there's no way
    to remotely call an arbitrary method like _check_grasp() on a worker's env.
    Rather than extend that protocol (which lives in LIBERO's own venv.py, not
    this project), computing this status *inside* step() and stashing it in
    info means it needs no new RPC at all: info is already round-tripped by
    the existing step() command."""

    def __init__(self, lift_height, container_distance, workspace_margin_xy=0.4, **env_kwargs):
        self.env = OffScreenRenderEnv(**env_kwargs)
        self.lift_height = lift_height
        self.container_distance = container_distance
        target_names = self.env.env.obj_of_interest
        self.container_name = find_container_name(target_names)
        self.pickable_names = find_pickable_names(target_names, self.container_name)
        self.initial_heights = {}
        self.workspace_bounds = get_workspace_bounds(self.env, margin_xy=workspace_margin_xy)

    def reset(self):
        obs = self.env.reset()
        self.initial_heights = target_body_heights(self.env, self.pickable_names)
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        info["touched"] = target_touched(self.env, self.pickable_names)
        info["grasped"] = target_grasped(self.env, self.pickable_names)
        info["lifted"] = target_lifted(self.env, self.pickable_names, self.initial_heights, self.lift_height)
        info["near_container"] = gripper_near_container(self.env, self.container_name, self.container_distance)
        info["object_near_container"] = object_near_container(
            self.env, self.pickable_names, self.container_name, self.container_distance,
        )
        info["out_of_workspace"] = gripper_out_of_workspace(obs, self.workspace_bounds)
        return obs, r, done, info

    def close(self):
        self.env.close()


def encode_clip_images_batch(clip_model, clip_preprocess, images, device):
    """images: list of N HWC uint8 numpy arrays (e.g. all N envs' agentview
    frames). Returns (N, clip_dim) embeddings from one batched forward pass."""
    batch = torch.stack([clip_preprocess(Image.fromarray(img)) for img in images]).to(device)
    with torch.no_grad():
        return clip_model.encode_image(batch).float()


def _query_policy_batch(policy, clip_model, clip_preprocess, obs_batch,
                         clip_hist, img_hist, proprio_hist, lang_batch, seq_len, replan_horizon, device):
    """Batched CLIP-encode + build_obs_features + history push/pad + frozen-base query +
    residual_head query for the M observations in obs_batch. clip_hist/img_hist/proprio_hist
    are lists of M per-item history lists, mutated in place; callers that must not disturb
    the original history should pass in copies. lang_batch: (M, lang_dim).

    Returns (decoded_last, base_actions_k, residual, value), each batched over M. Shared by
    collect_episodes_parallel's main rollout loop and its truncation-finalization pass so
    both paths build a policy query from a raw observation identically."""
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
    base_actions_k = base_actions[:, -1, :replan_horizon]                    # (M, replan_horizon, A)
    decoded_last = decoded.reshape(M, seq_len, *decoded.shape[1:])[:, -1]    # (M, chunk_size, hidden)

    residual, value = policy.residual_head(decoded_last, lang_batch)
    return decoded_last, base_actions_k, residual, value


@torch.no_grad()
def collect_episodes_parallel(policy, vec_env, lang_embeds, clip_model, clip_preprocess,
                               action_mean, action_std, device, max_raw_steps, seq_len,
                               touch_bonus=0.0, grasp_bonus=0.0, lift_bonus=0.0, release_bonus=0.0,
                               release_grace_steps=10, basket_penalty=0.0, basket_penalty_time_fraction=0.5,
                               out_of_frame_penalty=0.0, no_touch_penalty=0.0, replan_horizon=1):
    """Runs len(vec_env) episodes simultaneously, one per parallel environment,
    stepping the vectorized env in lockstep, batching the CLIP/policy forward
    passes across all N envs and running physics/rendering for all N envs
    across N subprocesses concurrently.

    A round's wall-clock cost is set by its slowest env, not the average, but
    still a large win over sequential collection. Envs that finish early keep
    getting stepped with a zero action until every
    env in the batch is done (matches LIBERO's own evaluate.py pattern; safe
    since stepping past done doesn't raise); no transition is recorded for an
    env once its own episode has ended. done_stack's done flag is LIBERO's own
    _check_success()-gated signal (bddl_base_domain.py), a true absorbing
    terminal state, not a horizon cutoff, which is what makes the bootstrap
    logic below correct (0 for done_stack, a real V(s) for the max_raw_steps
    cutoff).

    Returns a list of N episode dicts: decoded, base_action_k, action, mask,
    log_prob, value, reward, done, success, truncated, bootstrap_value."""
    N = len(vec_env)
    obs_list = vec_env.reset()

    clip_hist = [[] for _ in range(N)]
    img_hist = [[] for _ in range(N)]
    proprio_hist = [[] for _ in range(N)]
    touch_given = [False] * N
    grasp_given = [False] * N
    lift_given = [False] * N
    release_given = [False] * N
    basket_given = [False] * N
    out_of_frame_given = [False] * N
    episode_done = [False] * N
    raw_success = [False] * N
    steps_raw = [0] * N
    # Most recent raw step the target was actually grasped (not grasp_given, which stays
    # True forever after the first grasp). release_bonus needs recency: gating on
    # grasp_given alone would let the policy grasp briefly, drop the object elsewhere,
    # then swing an empty gripper near the basket for an unearned bonus.
    last_grasped_step = [-10**9] * N

    # GAE bootstrap value for the state right after each env's last recorded step. 0.0 is
    # already correct on true termination; only the max_raw_steps-truncation case needs
    # updating, computed in one finalization pass from a snapshot taken at the exact
    # moment of truncation (never from a stale live value, since a done env keeps getting
    # stepped with a zero action afterward).
    bootstrap_value = [0.0] * N
    done_via_truncation = [False] * N
    final_obs = [None] * N
    final_clip_hist = [None] * N
    final_img_hist = [None] * N
    final_proprio_hist = [None] * N

    decodeds = [[] for _ in range(N)]
    base_actions_k_list = [[] for _ in range(N)]
    actions_out = [[] for _ in range(N)]
    masks_out = [[] for _ in range(N)]
    log_probs_out = [[] for _ in range(N)]
    values_out = [[] for _ in range(N)]
    rewards_out = [[] for _ in range(N)]
    dones_out = [[] for _ in range(N)]

    lang_batch = torch.stack([le.squeeze(0) for le in lang_embeds]).to(device)  # (N, lang_dim)

    while not all(episode_done):
        decoded_last_batch, base_actions_k_batch, residual, value_batch = _query_policy_batch(
            policy, clip_model, clip_preprocess, obs_list, clip_hist, img_hist, proprio_hist,
            lang_batch, seq_len, replan_horizon, device,
        )

        k_per_env = [min(replan_horizon, max_raw_steps - steps_raw[i]) for i in range(N)]

        mean_k = base_actions_k_batch + residual[:, :replan_horizon]
        dist = torch.distributions.Normal(mean_k, policy.residual_head.std())
        sampled_batch = dist.sample()                                # (N, replan_horizon, A)
        log_prob_step = dist.log_prob(sampled_batch).sum(-1)          # (N, replan_horizon)
        actions_denorm = (sampled_batch * action_std + action_mean).cpu().numpy()

        macro_reward = [0.0] * N
        k_executed = [0] * N
        for j in range(replan_horizon):
            active = [i for i in range(N) if not episode_done[i] and j < k_per_env[i]]
            if not active:
                break
            step_actions = np.zeros((N,) + actions_denorm.shape[2:])
            for i in active:
                step_actions[i] = actions_denorm[i, j]
            obs_stack, rew_stack, done_stack, info_stack = vec_env.step(step_actions)
            obs_list = obs_stack
            for i in active:
                steps_raw[i] += 1
                k_executed[i] += 1
                r = float(rew_stack[i])
                raw_success[i] = raw_success[i] or (r > 0)
                macro_reward[i] += r
                info_i = info_stack[i]
                if touch_bonus > 0 and not touch_given[i] and info_i.get("touched"):
                    macro_reward[i] += touch_bonus
                    touch_given[i] = True
                if info_i.get("grasped"):
                    last_grasped_step[i] = steps_raw[i]
                if grasp_bonus > 0 and not grasp_given[i] and info_i.get("grasped"):
                    macro_reward[i] += grasp_bonus
                    grasp_given[i] = True
                if lift_bonus > 0 and not lift_given[i] and grasp_given[i] and info_i.get("lifted"):
                    macro_reward[i] += lift_bonus
                    lift_given[i] = True
                # One-time reward for opening the gripper while at the basket, having *recently*
                # grasped the target, a distinct failure mode from all the above: the arm
                # correctly carries the object to the basket but never releases it, so it never
                # actually drops in and the task never registers success. gripper_open reuses
                # build_obs_features' own threshold (robot0_gripper_qpos[0] > 0.03) rather than a
                # sim-level check, since it's already available on the raw obs dict with no need to
                # round-trip through GraspAwareEnv's info dict. Gated on recency (last_grasped_step
                # within release_grace_steps), not grasp_given (a one-time "ever grasped" flag that
                # stays True forever after the first grasp): grasp_given alone would let the policy
                # grasp briefly, drop the object somewhere else entirely, then swing an empty
                # gripper near the basket much later and collect a release bonus it never actually
                # earned. "Currently grasping" isn't usable directly either, that flips False the
                # instant the gripper opens, i.e. exactly the step this needs to detect.
                #
                # recently_grasped alone still isn't quite enough: last_grasped_step keeps updating
                # every step the object is actually held, so "grasp, carry the object somewhere
                # else, drop it, then rush an EMPTY gripper to the basket within release_grace_steps
                # and open" would still pass a recency-only check despite the object never coming
                # near the basket. object_near_container checks the object's own body position
                # (not the gripper's), closing that gap directly.
                gripper_open_i = obs_stack[i]["robot0_gripper_qpos"][0] > 0.03
                recently_grasped = (steps_raw[i] - last_grasped_step[i]) <= release_grace_steps
                if (release_bonus > 0 and not release_given[i] and recently_grasped
                        and gripper_open_i and info_i.get("near_container")
                        and info_i.get("object_near_container")):
                    macro_reward[i] += release_bonus
                    release_given[i] = True
                if (basket_penalty > 0 and not basket_given[i] and not grasp_given[i]
                        and steps_raw[i] <= basket_penalty_time_fraction * max_raw_steps
                        and info_i.get("near_container")):
                    macro_reward[i] -= basket_penalty
                    basket_given[i] = True
                if (out_of_frame_penalty > 0 and not out_of_frame_given[i]
                        and info_i.get("out_of_workspace")):
                    macro_reward[i] -= out_of_frame_penalty
                    out_of_frame_given[i] = True
                if done_stack[i] or steps_raw[i] >= max_raw_steps:
                    episode_done[i] = True
                    # One-time penalty, checked exactly once at episode end: never fires if the
                    # gripper touched a pickable target at any point (including a real success
                    # placing the object in the basket necessarily means it was touched first, so
                    # this is always a no-op there), so it targets pure non-engagement specifically
                    #, an arm that never even attempts contact, without the exploitable time
                    # window --basket_penalty had (a policy can't "wait out" an end-of-episode
                    # check the way it waited out a first-half-of-episode one; observed directly:
                    # the trained policy learned to approach the basket slowly enough that it only
                    # crossed --basket_penalty_distance after --basket_penalty_time_fraction had
                    # already elapsed, dodging that penalty while still exhibiting the exact
                    # never-touched-anything failure mode both penalties are meant to catch).
                    if no_touch_penalty > 0 and not touch_given[i]:
                        macro_reward[i] -= no_touch_penalty
                    if not done_stack[i]:
                        # max_raw_steps truncation, not a true terminal state, snapshot the
                        # observation and history window *right now*, at the exact moment of
                        # truncation. obs_stack[i] is a fresh dict returned by this call (never
                        # mutated by later vec_env.step() calls, which return entirely new dicts),
                        # so this stays correct even though (a) a done env still gets physically
                        # stepped with a zero action for the rest of this round's j-loop if
                        # replan_horizon > 1, and (b) clip_hist[i]/img_hist[i]/proprio_hist[i] keep
                        # getting mutated every remaining round of the whole while loop regardless
                        # of episode_done. Relying on those live, further-mutated structures (as an
                        # earlier version of this fix did, capturing opportunistically at the top of
                        # a later round) is only correct for replan_horizon == 1; snapshotting here
                        # is correct unconditionally.
                        done_via_truncation[i] = True
                        final_obs[i] = obs_stack[i]
                        final_clip_hist[i] = list(clip_hist[i])
                        final_img_hist[i] = list(img_hist[i])
                        final_proprio_hist[i] = list(proprio_hist[i])

        for i in range(N):
            if k_executed[i] == 0:
                continue  # this env's episode had already ended before this round started
            pad_len = replan_horizon - k_executed[i]
            action_padded = sampled_batch[i, :k_executed[i]].cpu()
            if pad_len > 0:
                action_padded = torch.cat([action_padded, torch.zeros(pad_len, action_padded.shape[-1])])
            mask = torch.cat([torch.ones(k_executed[i]), torch.zeros(pad_len)])

            decodeds[i].append(decoded_last_batch[i].cpu())
            base_actions_k_list[i].append(base_actions_k_batch[i].cpu())
            actions_out[i].append(action_padded)
            masks_out[i].append(mask)
            log_probs_out[i].append(log_prob_step[i, :k_executed[i]].sum().item())
            values_out[i].append(value_batch[i].item())
            rewards_out[i].append(macro_reward[i])
            dones_out[i].append(bool(episode_done[i]))

    # Finalization pass: every env that ended via max_raw_steps truncation needs one extra forward
    # pass (mirroring the loop's own feature-building) to get its correct bootstrap V(s), computed
    # from the observation/history *snapshots* taken at the exact moment of truncation above
    # never from the live obs_list/clip_hist/img_hist/proprio_hist, which keep changing for a done
    # env until the whole while loop exits (see the snapshot comment above).
    pending = [i for i in range(N) if done_via_truncation[i]]
    if pending:
        pending_obs = [final_obs[i] for i in pending]
        # Fresh copies: _query_policy_batch mutates its clip_hist/img_hist/proprio_hist args in
        # place, and these snapshots must stay untouched (nothing else reads them afterward, but
        # mutating someone else's "final_*" snapshot list on principle is asking for a future bug).
        pending_clip_hist = [list(final_clip_hist[i]) for i in pending]
        pending_img_hist = [list(final_img_hist[i]) for i in pending]
        pending_proprio_hist = [list(final_proprio_hist[i]) for i in pending]
        lang_pending = lang_batch[pending]

        _, _, _, value_pending = _query_policy_batch(
            policy, clip_model, clip_preprocess, pending_obs,
            pending_clip_hist, pending_img_hist, pending_proprio_hist,
            lang_pending, seq_len, replan_horizon, device,
        )
        for idx, i in enumerate(pending):
            bootstrap_value[i] = value_pending[idx].item()

    episodes = []
    for i in range(N):
        episodes.append({
            "decoded": decodeds[i], "base_action_k": base_actions_k_list[i],
            "action": actions_out[i], "mask": masks_out[i],
            "log_prob": log_probs_out[i], "value": values_out[i],
            "reward": rewards_out[i], "done": dones_out[i], "success": raw_success[i],
            "truncated": steps_raw[i] >= max_raw_steps and not raw_success[i],
            "bootstrap_value": bootstrap_value[i],
        })
    return episodes


def compute_gae(rewards, values, bootstrap_value, gamma, lam):
    """bootstrap_value is V(s) right after the last recorded step: 0.0 on true termination,
    or a real value estimate on max_raw_steps truncation. Since it already encodes the
    correct terminal condition, no separate done/truncated masking is needed."""
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    for t in reversed(range(T)):
        v_next = bootstrap_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update(policy, optimizer, batch, clip_eps, value_coef, entropy_coef,
                max_grad_norm, epochs, minibatch_size, device, target_kl=None, residual_l2_coef=0.0):
    """target_kl: if the approximate KL divergence between the old and current policy
    exceeds this after a minibatch, stop taking gradient steps on this batch and return
    early. Guards against a noisy small batch dragging the policy far in one update,
    since PPO's clip bounds the per-sample ratio but not aggregate drift over multiple
    epochs on the same batch. Uses the low-variance k3 estimator (joschu.net/blog/
    kl-approx.html). None disables the check.

    residual_l2_coef: weight on a penalty toward small residual magnitude (toward
    matching the frozen base). This guards cumulative drift across many updates, which
    target_kl (bounding only per-update drift) doesn't catch. 0 disables."""
    decoded = torch.stack(batch["decoded"]).to(device)
    base_action_k = torch.stack(batch["base_action_k"]).to(device)
    lang = torch.stack(batch["lang"]).to(device)
    actions = torch.stack(batch["action"]).to(device)
    mask = torch.stack(batch["mask"]).to(device)
    old_log_probs = torch.tensor(batch["log_prob"], dtype=torch.float32).to(device)
    advantages = torch.tensor(batch["advantage"], dtype=torch.float32).to(device)
    returns = torch.tensor(batch["return"], dtype=torch.float32).to(device)
    # unbiased=False (population std, divide by N not N-1): the population statistic is what
    # actually normalizes this specific, complete batch, Bessel's correction exists to reduce
    # bias when *estimating* a wider population's variance from a sample, which isn't what's
    # happening here. It also stays well-defined (0.0, not NaN) if a batch ever has exactly one
    # entry. Explicit zero-std guard below rather than relying on + 1e-8 alone: with the has_signal
    # gate upstream this is far less likely than it was, but a batch of near-identical returns
    # would otherwise still get its floating-point noise blown up into large fake advantages.
    adv_std = advantages.std(unbiased=False)
    if adv_std < 1e-8:
        advantages = torch.zeros_like(advantages)
    else:
        advantages = (advantages - advantages.mean()) / adv_std

    N = decoded.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "residual_l2": 0.0}
    n_batches = 0
    stopped_early = False
    for _ in range(epochs):
        if stopped_early:
            break
        idx = torch.randperm(N)
        for start in range(0, N, minibatch_size):
            mb = idx[start:start + minibatch_size]
            log_prob, entropy, value, residual_l2 = policy.evaluate_actions_from_cache(
                decoded[mb], base_action_k[mb], lang[mb], actions[mb], mask[mb],
            )

            log_ratio = log_prob - old_log_probs[mb]
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
            if target_kl is not None and approx_kl > target_kl:
                stopped_early = True
                stats["approx_kl"] = approx_kl
                break

            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            # Huber/smooth_l1 rather than raw MSE: quadratic near zero (same behavior as MSE for
            # typical-sized value errors) but linear beyond delta=1.0, so a handful of outlier
            # returns, plausible here given LIBERO's reward is terminal-only plus one-time
            # shaping bonuses, both of which produce occasional large jumps in return relative to
            # most transitions, can't dominate the value loss and drag the critic's gradient
            # around disproportionately the way a squared error would.
            value_loss = nn.functional.smooth_l1_loss(value, returns[mb])
            entropy_bonus = entropy.mean()
            residual_l2_bonus = residual_l2.mean()

            loss = (
                policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
                + residual_l2_coef * residual_l2_bonus
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in policy.residual_head.parameters() if p.requires_grad], max_grad_norm,
            )
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_bonus.item()
            stats["residual_l2"] += residual_l2_bonus.item()
            stats["approx_kl"] = approx_kl
            n_batches += 1

    stats["policy_loss"] /= max(n_batches, 1)
    stats["value_loss"] /= max(n_batches, 1)
    stats["entropy"] /= max(n_batches, 1)
    stats["residual_l2"] /= max(n_batches, 1)
    stats["stopped_early"] = stopped_early
    # How much of the intended epochs*minibatches budget target_kl actually let through, without
    # this, "stopped_early" alone can't distinguish an occasional safety-valve trip (most of the
    # budget used) from a systematic cap (a small, near-constant fraction every update).
    stats["n_batches"] = n_batches
    stats["total_batches"] = epochs * -(-N // minibatch_size)  # epochs * ceil(N / minibatch_size)
    return stats


def critic_warmup_update(policy, optimizer, batch, value_coef, epochs, minibatch_size, device):
    decoded = torch.stack(batch["decoded"]).to(device)
    base_action_k = torch.stack(batch["base_action_k"]).to(device)
    lang = torch.stack(batch["lang"]).to(device)
    mask = torch.stack(batch["mask"]).to(device)
    returns = torch.tensor(batch["return"], dtype=torch.float32).to(device)

    N = decoded.shape[0]
    total_loss = 0.0
    n_batches = 0
    for _ in range(epochs):
        idx = torch.randperm(N)
        for start in range(0, N, minibatch_size):
            mb = idx[start:start + minibatch_size]
            # values_only=True: only `value` is used below, so this skips constructing the
            # Normal distribution and summing log-probs/entropy/residual_l2 over it, pure
            # waste during warmup, which never needs any of the three. actions_batch is
            # unused on this path too, so no need to allocate the placeholder zeros tensor
            # the pre-values_only version had to pass here just to satisfy the signature.
            _, _, value, _ = policy.evaluate_actions_from_cache(
                decoded[mb], base_action_k[mb], lang[mb], None, mask[mb], values_only=True,
            )
            # Same Huber/smooth_l1 switch as ppo_update's actor-phase value loss, see there for
            # why; kept consistent across warmup and actor-phase critic fitting.
            loss = value_coef * nn.functional.smooth_l1_loss(value, returns[mb])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_eval(policy, tasks, task_ids, task_suite, camera_size, clip_model, clip_preprocess,
             action_mean, action_std, device, eval_trials, max_steps, seq_len,
             gif_dir=None, update=None, gif_trials=2):
    """Deterministic eval (policy.forward(), no sampling noise) on LIBERO's fixed
    init states, disjoint from the randomized env.reset() used for training
    rollouts, so this is a clean, comparable signal for checkpoint selection.

    Builds one eval env at a time, closing it before moving to the next task,
    rather than keeping all eval envs alive alongside the training envs. Keeping
    them all alive caused real resource contention in practice (eval got starved
    badly enough to produce results inconsistent with known baselines). A slower
    reliable eval beats a fast unreliable one.

    If gif_dir is given, saves a GIF for the first gif_trials trials of each task
    (capped independently of eval_trials, with all-tasks-by-default this would
    otherwise mean e.g. 10 tasks x 10 eval_trials = 100 GIFs every eval)."""
    was_training = policy.training
    policy.eval()
    update_dir = None
    if gif_dir is not None:
        update_dir = Path(gif_dir) / f"update_{update:04d}"
        update_dir.mkdir(parents=True, exist_ok=True)

    per_task_rates = []
    total_success, total_trials = 0, 0
    for tid in task_ids:
        info = tasks[tid]
        env = OffScreenRenderEnv(
            bddl_file_name=task_suite.get_task_bddl_file_path(tid),
            camera_heights=camera_size, camera_widths=camera_size,
            hard_reset=False,
        )
        try:
            init_states = load_init_states(info["task"])
            n_avail = init_states.shape[0]
            successes = 0
            for trial in range(eval_trials):
                init_state = init_states[trial % n_avail]
                want_frames = update_dir is not None and trial < gif_trials
                success, frames = rollout_episode(
                    policy, env, clip_model, clip_preprocess, info["lang_embed"],
                    action_mean, action_std, device, max_steps=max_steps, seq_len=seq_len,
                    collect_frames=want_frames, init_state=init_state,
                )
                successes += int(success)
                if want_frames and frames:
                    result = "success" if success else "fail"
                    gif_path = update_dir / f"task{tid}_trial{trial}_{result}.gif"
                    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        finally:
            env.close()
        rate = successes / eval_trials
        per_task_rates.append(rate)
        total_success += successes
        total_trials += eval_trials
        print(f"    eval task {tid:2d} [{info['language'][:45]:45s}] {successes}/{eval_trials} = {rate:.1%}")
    if was_training:
        policy.train()
    macro_rate = float(np.mean(per_task_rates))
    pooled_rate = total_success / total_trials
    return macro_rate, pooled_rate


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Namespaced so separate training invocations never clobber each other's checkpoints/GIFs
    # previously both landed in the same directory keyed only by update number, so a restarted
    # run would silently overwrite (or, worse, only partially overwrite) an earlier run's files
    # at the same update count.
    run_name = args.run_name or datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Run name: {run_name}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    policy = FiLMResidualPolicy(
        proprio_dim=bc_ckpt["proprio_dim"],
        chunk_size=bc_ckpt.get("chunk_size", 8),
        hidden_dim=bc_ckpt.get("hidden_dim", 256),
        init_log_std=args.init_log_std,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    ).to(device)
    policy.base.load_state_dict(bc_ckpt["policy_state_dict"])
    policy.freeze_base()
    assert 1 <= args.replan_horizon <= policy.chunk_size, (
        f"--replan_horizon ({args.replan_horizon}) must be >= 1 and can't exceed the BC "
        f"checkpoint's chunk_size ({policy.chunk_size})."
    )

    action_mean = torch.tensor(bc_ckpt["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(bc_ckpt["action_std"], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(policy.residual_head.parameters(), lr=args.lr)

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.suite != "libero_object":
        print(f"WARNING: --suite={args.suite!r}, find_container_name()'s destination-container "
              f"heuristic was only validated against libero_object. It may pick the wrong object "
              f"as the container on other suites, misdirecting the shaping bonuses (not the "
              f"terminal success reward). Worth spot-checking a few eval GIFs before trusting it.")
    task_ids = (
        [int(t) for t in args.task_ids.split(",")]
        if args.task_ids
        else list(range(task_suite.n_tasks))
    )
    tasks = {}
    for tid in task_ids:
        task = task_suite.get_task(tid)
        lang_embed = build_lang_embed(clip_model, task.language, device)
        tasks[tid] = {"lang_embed": lang_embed, "language": task.language, "task": task}
    print(f"Training on {len(task_ids)} task(s): {[tasks[t]['language'] for t in task_ids]}")

    # One persistent parallel env per task, each in its own subprocess.
    def make_env_fn(tid):
        def _fn():
            return GraspAwareEnv(
                lift_height=args.lift_height,
                container_distance=args.basket_penalty_distance,
                workspace_margin_xy=args.workspace_margin_xy,
                bddl_file_name=task_suite.get_task_bddl_file_path(tid),
                camera_heights=args.camera_size, camera_widths=args.camera_size,
                hard_reset=False,
            )
        return _fn

    # envs_per_task > 1 runs multiple parallel workers on the same task for more throughput.
    env_task_ids = [tid for tid in task_ids for _ in range(args.envs_per_task)]
    train_vec_env = SubprocVectorEnv([make_env_fn(tid) for tid in env_task_ids])
    train_lang_embeds = [tasks[tid]["lang_embed"] for tid in env_task_ids]
    # Ceiling division so a partial round still rounds up to meet episodes_per_update.
    rounds_per_update = max(1, -(-args.episodes_per_update // len(env_task_ids)))
    print(f"Parallel training envs: {len(env_task_ids)} ({args.envs_per_task}/task, "
          f"subprocess-parallel), {rounds_per_update} round(s)/update -> "
          f"{rounds_per_update * len(env_task_ids)} episodes/update")

    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_eval_rate = -1.0
    best_state_dict = None
    evals_without_improvement = 0
    # Counts only real actor updates. Critic-warmup and no-signal updates never touch
    # entropy_coef, so decaying it against the raw update index would burn through the
    # schedule before entropy is ever used.
    actor_updates_done = 0

    try:
        for update in range(args.updates):
            policy.eval()
            episodes = []
            for _ in range(rounds_per_update):
                batch_episodes = collect_episodes_parallel(
                    policy, train_vec_env, train_lang_embeds, clip_model, clip_preprocess,
                    action_mean, action_std, device, args.max_raw_steps, args.seq_len,
                    touch_bonus=args.touch_bonus, grasp_bonus=args.grasp_bonus, lift_bonus=args.lift_bonus,
                    release_bonus=args.release_bonus, release_grace_steps=args.release_grace_steps,
                    basket_penalty=args.basket_penalty, basket_penalty_time_fraction=args.basket_penalty_time_fraction,
                    out_of_frame_penalty=args.out_of_frame_penalty, no_touch_penalty=args.no_touch_penalty,
                    replan_horizon=args.replan_horizon,
                )
                for tid, ep in zip(env_task_ids, batch_episodes):
                    ep["lang_embed"] = tasks[tid]["lang_embed"]
                episodes.extend(batch_episodes)

            success_rate = np.mean([ep["success"] for ep in episodes])
            mean_return = np.mean([sum(ep["reward"]) for ep in episodes])

            batch = {"decoded": [], "base_action_k": [], "lang": [],
                     "action": [], "mask": [], "log_prob": [], "advantage": [], "return": []}
            for ep in episodes:
                adv, ret = compute_gae(ep["reward"], ep["value"], ep["bootstrap_value"],
                                        args.gamma, args.gae_lambda)
                n = len(ep["reward"])
                batch["decoded"].extend(ep["decoded"])
                batch["base_action_k"].extend(ep["base_action_k"])
                batch["lang"].extend([ep["lang_embed"].squeeze(0).cpu()] * n)
                batch["action"].extend(ep["action"])
                batch["mask"].extend(ep["mask"])
                batch["log_prob"].extend(ep["log_prob"])
                batch["advantage"].extend(adv)
                batch["return"].extend(ret)

            policy.train()
            # Advantage normalization divides by (std + 1e-8): when every episode in the batch
            # has zero return (as happens often under LIBERO's sparse terminal-only reward,
            # especially round-robinned thin across many tasks), the raw advantages are all
            # ~0 too, and normalizing blows tiny floating-point noise up into fake large
            # advantages. PPO then updates the actor on that noise, with nothing real to
            # anchor it, this randomly walks the residual away from the good BC-anchored
            # starting point instead of leaving it alone. So: only run the actor update on
            # batches that contain at least one real success; otherwise fall back to the
            # same critic-only fit used during warmup (correct thing to do either way, since
            # a value target of ~0 is still worth learning even with no successes yet).
            has_signal = any(ep["success"] for ep in episodes)
            if update < args.critic_warmup_updates or not has_signal:
                v_loss = critic_warmup_update(
                    policy, optimizer, batch, args.value_coef,
                    args.ppo_epochs, args.minibatch_size, device,
                )
                tag = "warmup" if update < args.critic_warmup_updates else "no-signal, skip actor"
                print(f"[{tag} {update+1}/{args.updates}] "
                      f"success={success_rate:.1%} return={mean_return:.3f} value_loss={v_loss:.4f}")
            else:
                # Exponential decay with a fixed half-life in actor-update count, floored at
                # --entropy_coef_final. An earlier version decayed linearly against a fraction
                # of total run length, which meant a longer run decayed proportionally slower
                # and let entropy climb unopposed for far too long.
                entropy_coef_now = max(
                    args.entropy_coef_final,
                    args.entropy_coef * (args.entropy_coef_decay_rate ** actor_updates_done),
                )
                # Same decay shape, opposite intent: loosens the anchor-to-base pressure once
                # training has real evidence about what's actually good.
                residual_l2_coef_now = max(
                    args.residual_l2_coef_final,
                    args.residual_l2_coef * (args.residual_l2_coef_decay_rate ** actor_updates_done),
                )
                stats = ppo_update(
                    policy, optimizer, batch, args.clip_eps, args.value_coef, entropy_coef_now,
                    args.max_grad_norm, args.ppo_epochs, args.minibatch_size, device,
                    target_kl=(args.target_kl if args.target_kl > 0 else None),
                    residual_l2_coef=residual_l2_coef_now,
                )
                actor_updates_done += 1
                early_tag = (
                    f" [target_kl early-stop {stats['n_batches']}/{stats['total_batches']} batches]"
                    if stats["stopped_early"] else ""
                )
                print(f"[update {update+1}/{args.updates}] success={success_rate:.1%} "
                      f"return={mean_return:.3f} policy_loss={stats['policy_loss']:.4f} "
                      f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
                      f"approx_kl={stats['approx_kl']:.4f} residual_l2={stats['residual_l2']:.4f}"
                      f"{early_tag}")

            do_eval = args.eval_every > 0 and ((update + 1) % args.eval_every == 0 or update == args.updates - 1)
            eval_rate = None
            if do_eval:
                print(f"  [eval @ update {update+1}] deterministic, {args.eval_trials} fixed-init trials/task:")
                eval_rate, pooled_rate = run_eval(
                    policy, tasks, task_ids, task_suite, args.camera_size, clip_model, clip_preprocess,
                    action_mean, action_std, device, args.eval_trials, args.max_raw_steps, args.seq_len,
                    gif_dir=(str(Path(args.eval_gif_dir) / run_name) if args.eval_gif_dir else None),
                    update=update + 1, gif_trials=args.eval_gif_trials,
                )
                print(f"  [eval @ update {update+1}] macro-avg success={eval_rate:.1%} pooled={pooled_rate:.1%}")

            do_save = (
                (args.save_every > 0 and (update + 1) % args.save_every == 0)
                or update == args.updates - 1
            )
            is_new_best = eval_rate is not None and eval_rate > best_eval_rate
            if do_save or is_new_best:
                ckpt = {
                    "policy_state_dict": policy.state_dict(),
                    "action_mean": bc_ckpt["action_mean"],
                    "action_std": bc_ckpt["action_std"],
                    "proprio_dim": bc_ckpt["proprio_dim"],
                    "chunk_size": bc_ckpt.get("chunk_size", 8),
                    "hidden_dim": bc_ckpt.get("hidden_dim", 256),
                    "log_std_min": args.log_std_min,
                    "log_std_max": args.log_std_max,
                    "suite": args.suite,
                    "task_ids": task_ids,
                    "bc_checkpoint": args.bc_checkpoint,
                    "eval_success_rate": eval_rate,
                }
                if do_save:
                    ckpt_path = save_dir / f"residual_ppo_seed{args.seed}_update{update+1}.pt"
                    torch.save(ckpt, ckpt_path)
                    print(f"  saved {ckpt_path}")
                if is_new_best:
                    best_eval_rate = eval_rate
                    best_state_dict = {k: v.detach().clone() for k, v in policy.residual_head.state_dict().items()}
                    best_path = save_dir / f"residual_ppo_seed{args.seed}_best.pt"
                    torch.save(ckpt, best_path)
                    print(f"  new best (eval success={eval_rate:.1%}) -> {best_path}")

            if eval_rate is not None:
                if is_new_best:
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    # Reload-on-regression: training can keep going well past its peak with
                    # nothing to stop it until --patience fires. Once eval falls meaningfully
                    # below the best ever seen, revert residual_head to that checkpoint and
                    # reset the optimizer (stale Adam momentum would otherwise keep pushing the
                    # reloaded weights back in the same bad direction). Only residual_head is
                    # touched; the frozen base never changes.
                    if (best_state_dict is not None and args.regression_reload_frac > 0
                            and best_eval_rate > 0 and eval_rate < args.regression_reload_frac * best_eval_rate):
                        policy.residual_head.load_state_dict(best_state_dict)
                        optimizer = torch.optim.Adam(policy.residual_head.parameters(), lr=args.lr)
                        # Reset patience too: the policy is back to its best state, so the
                        # evals that built up the counter no longer describe what's happening.
                        evals_without_improvement = 0
                        # Reset the entropy/residual_l2 decay clocks as well, so a run that keeps
                        # needing reloads keeps re-arming stronger anchoring each time.
                        actor_updates_done = 0
                        print(f"  reverted residual_head to best checkpoint (eval={eval_rate:.1%} < "
                              f"{args.regression_reload_frac:.0%} of best={best_eval_rate:.1%}); "
                              f"optimizer reset, patience counter reset, entropy/residual_l2 decay reset")
                    if args.patience > 0 and evals_without_improvement >= args.patience:
                        print(f"  early stopping: {evals_without_improvement} evals since last "
                              f"improvement (best={best_eval_rate:.1%}), patience={args.patience}")
                        break
    finally:
        train_vec_env.close()


if __name__ == "__main__":
    # Must happen before any CUDA init and before SubprocVectorEnv forks workers: Linux's
    # default "fork" start method silently hangs/crashes when forking a process that already
    # has a CUDA context. "forkserver" avoids that while staying fast, since the heavy
    # env-construction modules are preloaded once into the forkserver process itself.
    multiprocessing.set_forkserver_preload(["libero.libero.envs", "robosuite"])
    multiprocessing.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task_ids", type=str, default="",
                         help="comma-separated task ids to fine-tune on; default all tasks in --suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="",
                         help="namespaces checkpoints/GIFs under save_dir/run_name and "
                              "eval_gif_dir/run_name so separate runs never overwrite each other's "
                              "files; default auto-generates a timestamp")
    parser.add_argument("--camera_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=10, help="must match BC training seq_len")
    parser.add_argument("--max_raw_steps", type=int, default=300,
                         help="raw env steps before an episode is truncated (applies to both "
                              "training rollouts and eval). Close to LIBERO's own official 280 "
                              "for this suite.")
    parser.add_argument("--touch_bonus", type=float, default=0.12,
                         help="one-time training-only reward the first time any part of the gripper "
                              "contacts the target object (weaker than --grasp_bonus, which requires "
                              "robosuite's two-fingerpad pinch pattern). 0 disables. Kept below "
                              "--grasp_bonus so the staging (touch < grasp < lift < release < "
                              "terminal) still points toward full completion.")
    parser.add_argument("--grasp_bonus", type=float, default=0.15,
                         help="one-time training-only reward the first time the gripper grasps the "
                              "target (robosuite's contact-based _check_grasp), on top of LIBERO's "
                              "own terminal success reward. 0 disables. Kept well below the terminal "
                              "reward (1.0) so full task completion still dominates. Does not affect "
                              "eval or the reported success rate.")
    parser.add_argument("--lift_bonus", type=float, default=0.2,
                         help="one-time training-only reward the first time the target is raised "
                              "--lift_height off the table while grasped, on top of --grasp_bonus. "
                              "0 disables. Lift requires a real functional grasp, much harder to game "
                              "than a proximity-based bonus.")
    parser.add_argument("--lift_height", type=float, default=0.04,
                         help="meters above the object's starting height that counts as \"lifted\", "
                              "matches robosuite's own Lift task success threshold (lift.py).")
    parser.add_argument("--release_bonus", type=float, default=0.25,
                         help="one-time training-only reward the first time the gripper is open while "
                              "within --basket_penalty_distance of the destination container AND the "
                              "target object itself is also within that distance of the container, "
                              "gated on having grasped the target within the last "
                              "--release_grace_steps raw steps. 0 disables. Targets a distinct "
                              "failure mode from touch/grasp/lift: the arm carries the object to the "
                              "basket but never opens the gripper to release it.")
    parser.add_argument("--release_grace_steps", type=int, default=10,
                         help="--release_bonus only fires if the target was grasped within this many "
                              "raw steps before the release, not merely at any earlier point in the "
                              "episode. Gating on 'ever grasped' instead of recency would let the "
                              "policy grasp the target briefly, drop it elsewhere, then swing an "
                              "empty gripper near the basket much later and still collect the bonus. "
                              "--release_bonus's object_near_container check closes the remaining gap "
                              "by requiring the object's own position, not just the gripper's, be "
                              "near the container.")
    parser.add_argument("--basket_penalty", type=float, default=0.0,
                         help="one-time training-only penalty the first time the gripper gets within "
                              "--basket_penalty_distance of the destination container before ever "
                              "grasping the target, but only within the first "
                              "--basket_penalty_time_fraction of the episode. 0 disables (default): "
                              "gameable by approaching slowly enough to cross the distance threshold "
                              "only after the time-fraction gate closes. --no_touch_penalty supersedes "
                              "this with an end-of-episode check that has no such window to wait out.")
    parser.add_argument("--basket_penalty_distance", type=float, default=0.15,
                         help="meters from the destination container's body within which "
                              "--basket_penalty can fire (distance-based, not contact, hovering just "
                              "above the basket without touching it still counts).")
    parser.add_argument("--basket_penalty_time_fraction", type=float, default=0.5,
                         help="--basket_penalty only fires within this fraction of the episode "
                              "(default: first half). A late empty-handed basket visit, plausibly "
                              "after genuinely trying and failing to grasp, isn't penalized the same "
                              "as beelining there immediately.")
    parser.add_argument("--no_touch_penalty", type=float, default=0.3,
                         help="one-time training-only penalty applied once, at episode end, if the "
                              "gripper never touched a pickable target during the entire episode. "
                              "0 disables. Checked once at episode end rather than gated to part of "
                              "it, so there's no time window for the policy to learn to wait out.")
    parser.add_argument("--out_of_frame_penalty", type=float, default=0.15,
                         help="one-time training-only penalty the first time the end-effector strays "
                              "outside the table/workspace bounds (see --workspace_margin_xy). 0 "
                              "disables. Proprioception-based, no rendering needed. Targets the \"arm "
                              "swings out and disengages\" failure mode.")
    parser.add_argument("--workspace_margin_xy", type=float, default=0.4,
                         help="meters of horizontal margin added around the table's own footprint "
                              "before --out_of_frame_penalty can fire.")
    parser.add_argument("--replan_horizon", type=int, default=1,
                         help="raw env steps executed open-loop per policy query before replanning "
                              "(receding horizon); 1 = pure closed-loop, re-query every step. An "
                              "empirical sweep on this BC checkpoint showed success collapsing almost "
                              "immediately past k=1 (3/8, 0/8, 0/8, 0/8 for k=1..4), so this stays "
                              "closed-loop by default. Must be <= the BC checkpoint's chunk_size (8).")

    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--envs_per_task", type=int, default=2,
                         help="parallel subprocess workers per task, total simultaneous envs = "
                              "len(task_ids) * envs_per_task, e.g. 2 (default) with the standard "
                              "10-task suites runs 20 envs at once (2 episodes/task collected per "
                              "round instead of needing 2 sequential rounds). Raised from 1 to 2: "
                              "with the sparse terminal-only reward here, a 10-episode/update batch "
                              "gives GAE/PPO a high-variance advantage signal, and this doubles it to "
                              "~20/update for free at the same wall-clock cost per round. Each "
                              "additional worker is one more MuJoCo/GPU render context, so watch GPU "
                              "memory if you push this very high.")
    parser.add_argument("--episodes_per_update", type=int, default=10,
                         help="rounds_per_update = max(1, ceil(episodes_per_update / (len(task_ids) "
                              "* envs_per_task))); each round runs one episode per parallel env "
                              "simultaneously. Rounds up, so this is a floor on episodes actually "
                              "collected, not an exact target.")
    parser.add_argument("--critic_warmup_updates", type=int, default=20,
                         help="updates spent fitting only the value function before the actor trains, "
                              "so the actor doesn't update against inaccurate advantage estimates from "
                              "an untrained value network.")
    parser.add_argument("--eval_every", type=int, default=5,
                         help="run deterministic fixed-init-state eval every N updates, 0 disables")
    parser.add_argument("--eval_trials", type=int, default=10,
                         help="trials per task during periodic eval (kept small to stay cheap mid-training; "
                              "use benchmark_eval.py for the full 50-trial protocol on a final checkpoint)")
    parser.add_argument("--eval_gif_dir", type=str, default="eval_gifs_ppo",
                         help="dir to save eval-rollout GIFs each eval, pass '' to disable")
    parser.add_argument("--eval_gif_trials", type=int, default=1,
                         help="GIFs saved per task per eval, capped independently of --eval_trials "
                              "since disk usage adds up fast across many tasks and evals.")

    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Adam lr for the residual_head. Kept low since a higher lr caused a "
                              "fast early peak followed by a crash needing --regression_reload_frac.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                         help="starting entropy bonus coefficient; exponentially decayed toward "
                              "--entropy_coef_final at a fixed --entropy_coef_decay_rate per real "
                              "actor update (see both).")
    parser.add_argument("--entropy_coef_final", type=float, default=0.001,
                         help="floor for the exponential entropy-coefficient decay, never goes below "
                              "this, so exploration noise can't keep climbing indefinitely once "
                              "successes stop coming in.")
    parser.add_argument("--entropy_coef_decay_rate", type=float, default=0.985,
                         help="per-actor-update multiplicative decay rate for --entropy_coef, floored "
                              "at --entropy_coef_final. Default reaches roughly the floor by ~150 "
                              "actor updates. Fixed exponential decay, not linear against total "
                              "training length, so the half-life stays constant regardless of "
                              "--updates. 1.0 disables decay.")
    parser.add_argument("--target_kl", type=float, default=0.02,
                         help="stop taking further PPO gradient epochs on the current rollout batch "
                              "once the approximate KL divergence from the pre-update policy exceeds "
                              "this. 0 or negative disables. Guards against a noisy small batch "
                              "dragging the policy far off a peak in one update.")
    parser.add_argument("--residual_l2_coef", type=float, default=0.35,
                         help="starting weight on an L2 penalty toward small residual magnitude "
                              "(toward matching the frozen base); exponentially decayed toward "
                              "--residual_l2_coef_final at --residual_l2_coef_decay_rate per real "
                              "actor update. Unlike --target_kl (which only bounds drift within one "
                              "update), this bounds cumulative drift away from the base across many "
                              "updates. Decayed rather than constant, similar to Policy Decorator "
                              "(arxiv 2412.13630): stay close to the base early when there's no "
                              "evidence yet about what's good, loosen later for a well-earned "
                              "correction. 0 disables entirely.")
    parser.add_argument("--residual_l2_coef_final", type=float, default=0.035,
                         help="floor for the exponential --residual_l2_coef decay, so some anchoring "
                              "to the base always remains.")
    parser.add_argument("--residual_l2_coef_decay_rate", type=float, default=0.985,
                         help="per-actor-update multiplicative decay rate for --residual_l2_coef, "
                              "same reasoning as --entropy_coef_decay_rate. 1.0 disables decay.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--init_log_std", type=float, default=-2.0,
                         help="residual action std starts small (~0.14 in normalized action space) so early rollouts stay near the frozen BC policy")
    parser.add_argument("--log_std_min", type=float, default=-3.0,
                         help="floor on log_std, prevents the policy from collapsing to near-zero noise")
    parser.add_argument("--log_std_max", type=float, default=-1.0,
                         help="ceiling on log_std (~0.37 std in normalized action space). Prevents "
                              "the entropy bonus from inflating exploration noise without limit "
                              "when a run goes a long time with no reward signal to counterbalance it.")

    parser.add_argument("--save_dir", type=str, default="checkpoints_ppo")
    parser.add_argument("--save_every", type=int, default=10,
                         help="save a periodic (non-best) checkpoint every N updates; 0 disables "
                              "periodic saves (the final update is still always saved).")
    parser.add_argument("--patience", type=int, default=20,
                         help="stop after this many consecutive evals with no new best (0 disables). "
                              "--regression_reload_frac catches most peak-then-decline cases sooner; "
                              "this remains as a backstop in case a reload doesn't recover.")
    parser.add_argument("--regression_reload_frac", type=float, default=0.7,
                         help="if a non-improving eval's success rate falls below this fraction of "
                              "the best eval success seen so far, revert the residual_head's weights "
                              "to the best checkpoint and reset the optimizer, rather than continuing "
                              "to train from the regressed state. 0 disables. Directly targets the "
                              "peak-then-decline pattern --patience only stops (late, after wasting "
                              "compute) rather than prevents.")
    args = parser.parse_args()
    train(args)
