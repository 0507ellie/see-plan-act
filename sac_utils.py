"""Network and buffer building blocks for residual SAC fine-tuning (see train_sac.py
for the training loop). Hand-rolled on plain torch, same as train_ppo.py's PPO.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class _FiLMTrunk(nn.Module):
    """Shared state/task embedding: decoded[:, 0, :] (hidden_dim) + language (lang_dim)
    -> a film_hidden-dim task-conditioned feature, via FiLM modulation. Same pattern as
    train_ppo.py's FiLMResidualHead, plus a LayerNorm for Q-value stability. The actor
    and each Q-network build their own instance since they train on different losses
    and shouldn't share weights."""

    def __init__(self, hidden_dim, lang_dim, film_hidden=256):
        super().__init__()
        self.feature_proj = nn.Linear(hidden_dim, film_hidden)
        self.film_generator = nn.Linear(lang_dim, 2 * film_hidden)
        self.trunk = nn.Sequential(
            nn.Linear(film_hidden, film_hidden), nn.LayerNorm(film_hidden), nn.ReLU(),
        )

    def forward(self, decoded_step0, lang):
        h = self.feature_proj(decoded_step0)
        gamma, beta = self.film_generator(lang).chunk(2, dim=-1)
        h = gamma * h + beta
        return self.trunk(torch.relu(h))


class ResidualSACActor(nn.Module):
    """Tanh-squashed Gaussian residual actor, additive to the frozen base's chunk-step-0
    action: composed = base_action_0 + residual, residual = xi * tanh(u), u ~ N(mean, std).
    Only chunk-step-0 is corrected, matching train_ppo.py's finding that multi-step
    open-loop commitment collapses success on this BC checkpoint.

    xi hard-bounds |residual| in normalized action space (PLD's default of 0.5 for
    LIBERO). mean_head is zero-initialized so the actor starts identical to the frozen
    base; log_std_head's bias starts at init_log_std so exploration begins conservative."""

    def __init__(self, hidden_dim, lang_dim, action_dim, film_hidden=256, xi=0.5,
                 init_log_std=-2.0, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.xi = xi
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = _FiLMTrunk(hidden_dim, lang_dim, film_hidden)

        self.mean_head = nn.Linear(film_hidden, action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        self.log_std_head = nn.Linear(film_hidden, action_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, init_log_std)

    def forward(self, decoded_step0, lang):
        """-> (mean, log_std), both (N, action_dim), log_std already clamped."""
        h = self.trunk(decoded_step0, lang)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, decoded_step0, lang):
        """Reparameterized sample for the SAC actor loss, plus the deterministic residual.

        log_prob applies the standard tanh-squashing correction (Haarnoja et al. 2018,
        appendix C) extended by the constant xi scale factor: for a = xi*tanh(u),
        log p_A(a) = log p_U(u) - log(xi*(1-tanh(u)^2) + eps)."""
        mean, log_std = self.forward(decoded_step0, lang)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw = dist.rsample()
        tanh_raw = torch.tanh(raw)
        residual = self.xi * tanh_raw

        log_prob = dist.log_prob(raw) - torch.log(self.xi * (1 - tanh_raw.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1)

        deterministic_residual = self.xi * torch.tanh(mean)
        return residual, log_prob, deterministic_residual

    def act_deterministic(self, decoded_step0, lang):
        """No-sampling residual (xi*tanh(mean)), used at eval/rollout time."""
        mean, _ = self.forward(decoded_step0, lang)
        return self.xi * torch.tanh(mean)


class ResidualQNetwork(nn.Module):
    """Q(s, a): s is the same FiLM-modulated embedding the actor uses, a is the composed
    action actually executed (base_action_0 + residual), not the raw residual alone."""

    def __init__(self, hidden_dim, lang_dim, action_dim, film_hidden=256):
        super().__init__()
        self.trunk = _FiLMTrunk(hidden_dim, lang_dim, film_hidden)
        self.q_trunk = nn.Sequential(
            nn.Linear(film_hidden + action_dim, film_hidden), nn.LayerNorm(film_hidden), nn.ReLU(),
        )
        self.q_out = nn.Linear(film_hidden, 1)

    def forward(self, decoded_step0, lang, action):
        h = self.trunk(decoded_step0, lang)
        h = torch.cat([h, action], dim=-1)
        h = self.q_trunk(h)
        return self.q_out(h).squeeze(-1)


class TwinQCritic(nn.Module):
    """Clipped double-Q (Fujimoto et al. 2018): two independent Q-networks, the SAC
    update takes min(q1, q2) to counteract overestimation."""

    def __init__(self, hidden_dim, lang_dim, action_dim, film_hidden=256):
        super().__init__()
        self.q1 = ResidualQNetwork(hidden_dim, lang_dim, action_dim, film_hidden)
        self.q2 = ResidualQNetwork(hidden_dim, lang_dim, action_dim, film_hidden)

    def forward(self, decoded_step0, lang, action):
        return self.q1(decoded_step0, lang, action), self.q2(decoded_step0, lang, action)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging: target <- tau*source + (1-tau)*target, in place."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class ReplayBuffer:
    """Fixed-capacity ring buffer of single-step transitions for one task specialist's
    SAC training. Stores the frozen base's cached decoded[:, 0, :] feature (fp16) instead
    of raw images/CLIP embeddings, so the backbone never gets re-run on a sampled
    minibatch. task_id is stored instead of the full language vector; train_sac.py
    resolves it against a {task_id: lang_embed} dict at sample time."""

    def __init__(self, capacity, hidden_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.decoded = torch.zeros(capacity, hidden_dim, dtype=torch.float16)
        self.next_decoded = torch.zeros(capacity, hidden_dim, dtype=torch.float16)
        self.base_action = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.next_base_action = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.action = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.reward = torch.zeros(capacity, dtype=torch.float32)
        self.done = torch.zeros(capacity, dtype=torch.bool)
        self.task_id = torch.zeros(capacity, dtype=torch.int8)
        self._ptr = 0
        self._size = 0

    @staticmethod
    def _to_cpu(x, dtype):
        # Inputs are usually CUDA tensors straight out of the frozen base's forward pass;
        # this buffer lives entirely on CPU, so everything gets detached and moved here.
        return torch.as_tensor(x).detach().to(dtype=dtype, device="cpu")

    def add(self, decoded, next_decoded, base_action, next_base_action, action, reward,
            done, task_id):
        """One transition. Tensor args are 1D (CPU or CUDA)."""
        i = self._ptr
        self.decoded[i] = self._to_cpu(decoded, torch.float16)
        self.next_decoded[i] = self._to_cpu(next_decoded, torch.float16)
        self.base_action[i] = self._to_cpu(base_action, torch.float32)
        self.next_base_action[i] = self._to_cpu(next_base_action, torch.float32)
        self.action[i] = self._to_cpu(action, torch.float32)
        self.reward[i] = float(reward)
        self.done[i] = bool(done)
        self.task_id[i] = int(task_id)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self._size, (batch_size,))
        return {
            "decoded": self.decoded[idx].float().to(self.device),
            "next_decoded": self.next_decoded[idx].float().to(self.device),
            "base_action": self.base_action[idx].to(self.device),
            "next_base_action": self.next_base_action[idx].to(self.device),
            "action": self.action[idx].to(self.device),
            "reward": self.reward[idx].to(self.device),
            "done": self.done[idx].to(self.device),
            "task_id": self.task_id[idx].to(self.device),
        }

    def __len__(self):
        return self._size
