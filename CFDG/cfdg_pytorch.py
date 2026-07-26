"""Single-file PyTorch reproduction of IQL + CFDG.

Paper:
    Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion
    Generation (ICML 2025).

CFDG is a data-augmentation framework rather than a standalone policy optimizer.
The paper integrates it with IQL, PEX, and APL. This file implements the IQL +
CFDG path, which is also the base used in the paper's direct comparisons with
SynthER and EDIS.

Paper-specified CFDG components reproduced here:

1. Offline and online transitions are treated as two separate class labels.
2. A single conditional/unconditional denoising network is jointly trained by
   randomly replacing the class label with a null label.
3. Classifier-free guidance uses
       D_cfg = (1 + w) D_cond - w D_uncond.
4. The diffusion model is an EDM-style model with a residual MLP denoiser,
   depth 6, width 1024, ReLU activations, Adam at 3e-4, cosine annealing,
   batch size 256, 100,000 training steps per refresh, and 128 denoising steps.
5. During online fine-tuning the diffusion model is refreshed every 100,000
   environment steps for IQL, using both the fixed offline buffer and the
   growing online buffer.
6. Separate offline-synthetic and online-synthetic replay buffers are used.
7. Synthetic data occupies one third of each policy-learning batch. The real
   offline, real online, and synthetic proportions are therefore 1:1:1.
8. Inside the synthetic portion, online-synthetic to offline-synthetic data is
   sampled at the paper's 8:2 ratio.
9. The combined synthetic replay capacity is 1,000,000 transitions.
10. IQL is pretrained for 1,000,000 updates and fine-tuned for 1,000,000 online
    environment steps, following the paper's IQL setting.
11. The IQL base follows the public PyTorch PEX implementation: AntMaze reward
    shifting, a state-independent policy standard deviation, offline actor-LR
    cosine annealing, 5,000 initial online collection steps, and fresh online
    optimizers initialized from the offline-trained networks.

The paper does not publish a verifiable author CFDG repository and does not
specify several low-level choices, including transition normalization,
classifier-free label-drop probability, guidance strength, EMA decay, the
training mixture of offline and online diffusion samples, or the number of
samples generated at each refresh. These are explicit configuration
fields below rather than being presented as author-specified constants. The
sampling implementation follows the stochastic EDM/Heun sampler of Karras et
al. and uses its ImageNet defaults where the paper explicitly requests them.

PyTorch implementations cannot be bitwise identical to another framework or to
unreleased author code. This file is a paper-faithful, auditable reproduction,
not a claim of numerical identity with unavailable source code.
"""

import math
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import d4rl  # noqa: F401: registers D4RL Gym environments
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


NumpyBatch = Dict[str, np.ndarray]
TorchBatch = Dict[str, torch.Tensor]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainConfig:
    # Paper experiments cover MuJoCo Locomotion, AntMaze, and an Adroit appendix.
    env_name: str = "halfcheetah-medium-replay-v2"

    # Standard IQL configuration used by the public PyTorch IQL/PEX code family.
    hidden_dim: int = 256
    hidden_layers: int = 2
    actor_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 5e-3
    expectile: Optional[float] = None
    advantage_temperature: Optional[float] = None
    max_advantage_weight: float = 100.0
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 2.0

    batch_size: int = 256
    pretrain_steps: int = 1_000_000
    num_total_steps: int = 1_000_000
    online_updates_per_step: int = 1
    online_replay_size: int = 1_000_000

    # Paper-specified CFDG module settings (Table 5 and Section 4.1).
    diffusion_hidden_dim: int = 1024
    diffusion_depth: int = 6
    diffusion_learning_rate: float = 3e-4
    diffusion_batch_size: int = 256
    diffusion_train_steps: int = 100_000
    diffusion_refresh_every: int = 100_000  # IQL and PEX setting
    diffusion_sampling_steps: int = 128
    synthetic_buffer_size: int = 1_000_000  # combined off-syn + on-syn
    synthetic_batch_ratio: float = 1.0 / 3.0
    online_synthetic_fraction: float = 0.8  # online-syn : offline-syn = 8 : 2
    diffusion_online_fraction: float = 0.5  # paper does not specify this ratio

    # The paper does not state these low-level values. They are exposed rather
    # than silently being attributed to the authors.
    classifier_free_drop_probability: float = 0.1
    classifier_free_guidance_weight: float = 1.0
    diffusion_ema_decay: float = 0.995
    diffusion_sigma_data: float = 0.5
    diffusion_p_mean: float = -1.2
    diffusion_p_std: float = 1.2

    # EDM stochastic sampler defaults reported by Karras et al. for ImageNet.
    edm_sigma_min: float = 0.002
    edm_sigma_max: float = 80.0
    edm_rho: float = 7.0
    edm_s_churn: float = 40.0
    edm_s_min: float = 0.05
    edm_s_max: float = 50.0
    edm_s_noise: float = 1.003

    # Generation is spread over refreshes so that the combined buffer reaches
    # approximately one million samples by the end of online fine-tuning.
    synthetic_samples_per_refresh: Optional[int] = None
    generation_batch_size: int = 2048

    # Normalize full transition vectors using fixed offline-data statistics.
    normalize_diffusion_transitions: bool = True
    diffusion_normalization_eps: float = 1e-6
    generated_mask_threshold: float = 0.5

    start_training: int = 5_000
    eval_episodes: int = 10
    eval_every: int = 5_000
    log_every: int = 1_000

    train_seed: int = 42
    eval_seed_offset: int = 42
    deterministic_torch: bool = False

    checkpoints_path: Optional[str] = None
    log_root: str = "logs/CFDG_IQL_PyTorch"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Environment compatibility
# =============================================================================


def _convert_observation_to_float32(observation):
    if isinstance(observation, np.ndarray) and observation.dtype == np.float64:
        return observation.astype(np.float32)
    if isinstance(observation, dict):
        return {
            key: _convert_observation_to_float32(value)
            for key, value in observation.items()
        }
    return observation


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if isinstance(space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=np.float32,
            )
        elif isinstance(space, gym.spaces.Dict):
            converted = {}
            for key, subspace in space.spaces.items():
                if not isinstance(subspace, gym.spaces.Box):
                    raise NotImplementedError(
                        "Only Box entries are supported in Dict observations."
                    )
                converted[key] = gym.spaces.Box(
                    low=subspace.low,
                    high=subspace.high,
                    shape=subspace.shape,
                    dtype=np.float32,
                )
            self.observation_space = gym.spaces.Dict(converted)
        else:
            raise NotImplementedError(
                f"Unsupported observation space: {type(space).__name__}"
            )

    def observation(self, observation):
        return _convert_observation_to_float32(observation)


class UniversalSeed(gym.Wrapper):
    def seed(self, seed: int):
        seeds = None
        try:
            seeds = self.env.seed(seed)
        except Exception:
            pass
        try:
            self.env.observation_space.seed(seed)
        except Exception:
            pass
        try:
            self.env.action_space.seed(seed)
        except Exception:
            pass
        return seeds


def wrap_gym(env: gym.Env) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


def _seed_env(env: gym.Env, seed: int) -> None:
    try:
        env.seed(seed)
    except Exception:
        try:
            env.reset(seed=seed)
        except Exception:
            pass
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass


def _reset_env(env: gym.Env, seed: Optional[int] = None):
    try:
        output = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None:
            _seed_env(env, seed)
        output = env.reset()
    return output[0] if isinstance(output, tuple) else output


def _step_env(env: gym.Env, action):
    output = env.step(action)
    if len(output) == 5:
        next_observation, reward, terminated, truncated, info = output
        done = bool(terminated or truncated)
        info = dict(info)
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True
        return next_observation, float(reward), done, info
    next_observation, reward, done, info = output
    return next_observation, float(reward), bool(done), dict(info)


# =============================================================================
# Datasets and replay buffers
# =============================================================================


class ArrayDataset:
    def __init__(self, arrays: Mapping[str, np.ndarray], seed: int):
        required = {
            "observations",
            "actions",
            "rewards",
            "masks",
            "next_observations",
        }
        missing = required.difference(arrays)
        if missing:
            raise KeyError(f"Dataset is missing fields: {sorted(missing)}")
        lengths = {len(np.asarray(value)) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("Dataset fields have inconsistent lengths.")
        self.arrays = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in arrays.items()
            if key in required
        }
        self.size = lengths.pop()
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size < 1:
            raise ValueError("Cannot sample from an empty dataset.")
        indices = self.rng.integers(self.size, size=batch_size)
        return {key: value[indices] for key, value in self.arrays.items()}


def _transform_reward(reward, env_name: str):
    """Match the public PyTorch IQL/PEX data preprocessing.

    AntMaze rewards are shifted from {0, 1} to {-1, 0}; MuJoCo locomotion
    rewards are left unchanged. The same transform is applied to offline and
    newly collected online transitions.
    """
    if "antmaze" in env_name.lower():
        return np.asarray(reward, dtype=np.float32) - 1.0
    return np.asarray(reward, dtype=np.float32)


class D4RLDataset(ArrayDataset):
    def __init__(self, env: gym.Env, env_name: str, seed: int):
        data = d4rl.qlearning_dataset(env)
        terminals = np.asarray(data["terminals"], dtype=np.float32)
        arrays = {
            "observations": np.asarray(data["observations"], dtype=np.float32),
            "actions": np.asarray(data["actions"], dtype=np.float32),
            "rewards": _transform_reward(data["rewards"], env_name).reshape(-1),
            "masks": 1.0 - terminals.reshape(-1),
            "next_observations": np.asarray(
                data["next_observations"], dtype=np.float32
            ),
        }
        super().__init__(arrays, seed=seed)


class ReplayBuffer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
    ):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        next_observation: np.ndarray,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.masks[index] = mask
        self.next_observations[index] = next_observation
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, batch: NumpyBatch) -> None:
        count = len(batch["rewards"])
        if count == 0:
            return
        if count >= self.capacity:
            start = count - self.capacity
            self.observations[:] = batch["observations"][start:]
            self.actions[:] = batch["actions"][start:]
            self.rewards[:] = batch["rewards"][start:]
            self.masks[:] = batch["masks"][start:]
            self.next_observations[:] = batch["next_observations"][start:]
            self.size = self.capacity
            self.position = 0
            return
        first = min(count, self.capacity - self.position)
        second = count - first
        sl = slice(self.position, self.position + first)
        self.observations[sl] = batch["observations"][:first]
        self.actions[sl] = batch["actions"][:first]
        self.rewards[sl] = batch["rewards"][:first]
        self.masks[sl] = batch["masks"][:first]
        self.next_observations[sl] = batch["next_observations"][:first]
        if second > 0:
            sl2 = slice(0, second)
            self.observations[sl2] = batch["observations"][first:]
            self.actions[sl2] = batch["actions"][first:]
            self.rewards[sl2] = batch["rewards"][first:]
            self.masks[sl2] = batch["masks"][first:]
            self.next_observations[sl2] = batch["next_observations"][first:]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size < 1:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = self.rng.integers(self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "next_observations": self.next_observations[indices],
        }


# =============================================================================
# IQL networks and learner
# =============================================================================


def _init_linear(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """PyTorch-default MLP used by the public PEX IQL critic and value nets."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        modules = []
        previous = input_dim
        for hidden in hidden_dims:
            modules.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        modules.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class GaussianPolicy(nn.Module):
    """Gaussian policy matching the public PyTorch IQL/PEX implementation.

    The mean is mapped into the action range with tanh, while the standard
    deviation is a state-independent trainable vector. The Gaussian itself is
    not additionally tanh-transformed, matching ``scale_distribution=False``
    and ``state_dependent_std=False`` in the PEX codebase.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("GaussianPolicy requires at least one hidden layer.")

        modules = []
        previous = observation_dim
        for hidden in hidden_dims:
            modules.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        self.net = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(previous, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        self.register_buffer(
            "action_mean",
            torch.as_tensor((action_high + action_low) / 2.0),
        )
        self.register_buffer(
            "action_magnitude",
            torch.as_tensor((action_high - action_low) / 2.0),
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # PEX applies Xavier initialization only to the policy's Linear layers.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def _distribution_parameters(self, observations: torch.Tensor):
        features = self.net(observations)
        mean = self.action_mean + self.action_magnitude * torch.tanh(
            self.mean_linear(features)
        )
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def log_prob(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_std = self._distribution_parameters(observations)
        return Normal(mean, log_std.exp()).log_prob(actions).sum(dim=-1)

    def sample(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution_parameters(observations)
        distribution = Normal(mean, log_std.exp())
        actions = mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        return actions, log_prob

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        device: torch.device,
        deterministic: bool,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action, _ = self.sample(tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.net = MLP(observation_dim + action_dim, 1, hidden_dims)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([observations, actions], dim=-1)).squeeze(-1)


class DoubleCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dims)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observations, actions), self.q2(observations, actions)

    def minimum(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self(observations, actions)
        return torch.minimum(q1, q2)


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.net = MLP(observation_dim, 1, hidden_dims)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


def _expectile_loss(
    difference: torch.Tensor,
    expectile: float,
) -> torch.Tensor:
    weight = torch.where(
        difference > 0.0,
        torch.as_tensor(expectile, device=difference.device),
        torch.as_tensor(1.0 - expectile, device=difference.device),
    )
    return weight * difference.square()


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


class IQLLearner:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: TrainConfig,
        device: torch.device,
    ):
        hidden_dims = tuple(
            config.hidden_dim for _ in range(config.hidden_layers)
        )
        self.actor = GaussianPolicy(
            observation_dim,
            action_dim,
            hidden_dims,
            action_low,
            action_high,
            config.actor_log_std_min,
            config.actor_log_std_max,
        ).to(device)
        self.critic = DoubleCritic(
            observation_dim, action_dim, hidden_dims
        ).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.target_critic.requires_grad_(False)
        self.value = ValueNetwork(observation_dim, hidden_dims).to(device)

        self.discount = float(config.discount)
        self.tau = float(config.tau)
        self.expectile = float(config.expectile)
        self.advantage_temperature = float(config.advantage_temperature)
        self.max_advantage_weight = float(config.max_advantage_weight)
        self.config = config
        self.device = device

        self._reset_optimizers(offline_phase=True)

    def _reset_optimizers(self, offline_phase: bool) -> None:
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=self.config.value_learning_rate
        )
        self.actor_scheduler = None
        if offline_phase:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=max(1, self.config.pretrain_steps),
            )

    def start_online_phase(self) -> None:
        """Match PEX's fresh online IQL learner initialized from offline nets."""
        self.target_critic.load_state_dict(self.critic.state_dict())
        self._reset_optimizers(offline_phase=False)

    def update(
        self,
        batch: TorchBatch,
        offline_phase: bool = False,
    ) -> Dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]

        # PEX IQL computes both quantities before any parameter update.
        with torch.no_grad():
            target_q = self.target_critic.minimum(observations, actions)
            next_values = self.value(batch["next_observations"])

        values = self.value(observations)
        advantages = target_q.detach() - values
        value_loss = _expectile_loss(advantages, self.expectile).mean()

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            bellman_target = (
                batch["rewards"]
                + self.discount * batch["masks"] * next_values.detach()
            )
        q1, q2 = self.critic(observations, actions)
        critic_loss = 0.5 * (
            F.mse_loss(q1, bellman_target)
            + F.mse_loss(q2, bellman_target)
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # The public PEX IQL target update precedes the policy update.
        _soft_update(self.target_critic, self.critic, self.tau)

        with torch.no_grad():
            exponential_advantage = torch.exp(
                self.advantage_temperature * advantages.detach()
            ).clamp(max=self.max_advantage_weight)
        log_probability = self.actor.log_prob(observations, actions.detach())
        actor_loss = -(exponential_advantage * log_probability).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        if offline_phase and self.actor_scheduler is not None:
            self.actor_scheduler.step()

        return {
            "value_loss": float(value_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "value": float(values.detach().mean().item()),
            "q": float(target_q.detach().mean().item()),
            "advantage_weight": float(
                exponential_advantage.detach().mean().item()
            ),
            "actor_learning_rate": float(
                self.actor_optimizer.param_groups[0]["lr"]
            ),
        }


# =============================================================================
# Transition codec and CFDG diffusion model
# =============================================================================


class TransitionCodec:
    """Packs complete transitions into the vector modeled by CFDG."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        normalize: bool,
        eps: float,
        mask_threshold: float,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = 2 * observation_dim + action_dim + 2
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.normalize_enabled = bool(normalize)
        self.eps = float(eps)
        self.mask_threshold = float(mask_threshold)
        self.mean = np.zeros((self.transition_dim,), dtype=np.float32)
        self.std = np.ones((self.transition_dim,), dtype=np.float32)

    def pack_numpy(self, batch: NumpyBatch) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(batch["observations"], dtype=np.float32),
                np.asarray(batch["actions"], dtype=np.float32),
                np.asarray(batch["rewards"], dtype=np.float32).reshape(-1, 1),
                np.asarray(batch["masks"], dtype=np.float32).reshape(-1, 1),
                np.asarray(batch["next_observations"], dtype=np.float32),
            ],
            axis=-1,
        )

    def fit(self, dataset: ArrayDataset) -> None:
        vectors = self.pack_numpy(dataset.arrays)
        if self.normalize_enabled:
            self.mean = vectors.mean(axis=0).astype(np.float32)
            self.std = vectors.std(axis=0).astype(np.float32)
            self.std = np.maximum(self.std, self.eps).astype(np.float32)
        else:
            self.mean.fill(0.0)
            self.std.fill(1.0)

    def normalize_numpy(self, vectors: np.ndarray) -> np.ndarray:
        return ((vectors - self.mean) / self.std).astype(np.float32)

    def denormalize_tensor(self, vectors: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=vectors.dtype, device=vectors.device)
        std = torch.as_tensor(self.std, dtype=vectors.dtype, device=vectors.device)
        return vectors * std + mean

    def unpack_tensor(self, vectors: torch.Tensor) -> NumpyBatch:
        vectors = self.denormalize_tensor(vectors)
        o0 = 0
        o1 = self.observation_dim
        a1 = o1 + self.action_dim
        r1 = a1 + 1
        m1 = r1 + 1
        observations = vectors[:, o0:o1]
        actions = vectors[:, o1:a1]
        rewards = vectors[:, a1:r1].squeeze(-1)
        masks = vectors[:, r1:m1].squeeze(-1)
        next_observations = vectors[:, m1:]

        action_low = torch.as_tensor(
            self.action_low, dtype=actions.dtype, device=actions.device
        )
        action_high = torch.as_tensor(
            self.action_high, dtype=actions.dtype, device=actions.device
        )
        actions = torch.maximum(torch.minimum(actions, action_high), action_low)
        masks = (masks >= self.mask_threshold).to(torch.float32)

        return {
            "observations": observations.detach().cpu().numpy().astype(np.float32),
            "actions": actions.detach().cpu().numpy().astype(np.float32),
            "rewards": rewards.detach().cpu().numpy().astype(np.float32),
            "masks": masks.detach().cpu().numpy().astype(np.float32),
            "next_observations": next_observations.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        }


class FourierFeatures(nn.Module):
    def __init__(self, output_dim: int, scale: float = 16.0):
        super().__init__()
        if output_dim % 2 != 0:
            raise ValueError("Fourier feature dimension must be even.")
        frequencies = torch.randn(output_dim // 2) * scale
        self.register_buffer("frequencies", frequencies)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * inputs.unsqueeze(-1) * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, conditioning_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.condition = nn.Linear(conditioning_dim, hidden_dim)
        _init_linear(self.linear1, gain=math.sqrt(2.0))
        _init_linear(self.linear2, gain=1e-2)
        _init_linear(self.condition, gain=1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.linear1(inputs) + self.condition(conditioning)
        hidden = F.relu(hidden)
        hidden = self.linear2(hidden)
        return F.relu(inputs + hidden)


class ConditionalResidualMLP(nn.Module):
    """Paper-specified 6-layer residual MLP denoising network."""

    def __init__(
        self,
        transition_dim: int,
        hidden_dim: int,
        depth: int,
        num_classes_with_null: int = 3,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.noise_embedding = FourierFeatures(embedding_dim)
        self.class_embedding = nn.Embedding(
            num_classes_with_null, embedding_dim
        )
        self.conditioning = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.input_layer = nn.Linear(transition_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(hidden_dim, transition_dim)

        _init_linear(self.input_layer, gain=math.sqrt(2.0))
        _init_linear(self.output_layer, gain=1e-2)
        for module in self.conditioning:
            if isinstance(module, nn.Linear):
                _init_linear(module, gain=math.sqrt(2.0))
        nn.init.normal_(self.class_embedding.weight, std=0.02)

    def forward(
        self,
        inputs: torch.Tensor,
        noise_condition: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        noise_features = self.noise_embedding(noise_condition)
        class_features = self.class_embedding(labels)
        condition = self.conditioning(
            torch.cat([noise_features, class_features], dim=-1)
        )
        hidden = F.relu(self.input_layer(inputs))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output_layer(hidden)


class EDMPreconditionedModel(nn.Module):
    def __init__(self, denoiser: nn.Module, sigma_data: float):
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = float(sigma_data)

    def forward(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1)
        sigma_data = self.sigma_data
        denominator = sigma.square() + sigma_data**2
        c_skip = sigma_data**2 / denominator
        c_out = sigma * sigma_data / torch.sqrt(denominator)
        c_in = torch.rsqrt(denominator)
        c_noise = torch.log(sigma.clamp_min(1e-8)).squeeze(-1) / 4.0
        residual = self.denoiser(c_in * noisy, c_noise, labels)
        return c_skip * noisy + c_out * residual


class CFDGModel:
    OFFLINE_LABEL = 0
    ONLINE_LABEL = 1
    NULL_LABEL = 2

    def __init__(
        self,
        codec: TransitionCodec,
        config: TrainConfig,
        device: torch.device,
    ):
        denoiser = ConditionalResidualMLP(
            codec.transition_dim,
            config.diffusion_hidden_dim,
            config.diffusion_depth,
        )
        self.model = EDMPreconditionedModel(
            denoiser, config.diffusion_sigma_data
        ).to(device)
        self.ema_model = deepcopy(self.model).to(device)
        self.ema_model.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.diffusion_learning_rate
        )
        self.codec = codec
        self.config = config
        self.device = device
        self.train_updates = 0

    def _sample_training_vectors(
        self,
        offline_dataset: ArrayDataset,
        online_buffer: ReplayBuffer,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if online_buffer.size < 1:
            raise ValueError("CFDG training requires non-empty online data.")
        online_count = int(round(
            batch_size * self.config.diffusion_online_fraction
        ))
        online_count = min(max(1, online_count), batch_size - 1)
        offline_count = batch_size - online_count
        offline = self.codec.normalize_numpy(
            self.codec.pack_numpy(offline_dataset.sample(offline_count))
        )
        online = self.codec.normalize_numpy(
            self.codec.pack_numpy(online_buffer.sample(online_count))
        )
        vectors = np.concatenate([offline, online], axis=0)
        labels = np.concatenate(
            [
                np.full(offline_count, self.OFFLINE_LABEL, dtype=np.int64),
                np.full(online_count, self.ONLINE_LABEL, dtype=np.int64),
            ],
            axis=0,
        )
        permutation = np.random.permutation(batch_size)
        vectors = torch.as_tensor(
            vectors[permutation], dtype=torch.float32, device=self.device
        )
        labels = torch.as_tensor(
            labels[permutation], dtype=torch.long, device=self.device
        )
        return vectors, labels

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.config.diffusion_ema_decay
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)

    def train_refresh(
        self,
        offline_dataset: ArrayDataset,
        online_buffer: ReplayBuffer,
        writer: Optional[SummaryWriter],
        online_step: int,
    ) -> Dict[str, float]:
        self.model.train()
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.config.diffusion_learning_rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.diffusion_train_steps),
        )
        final_loss = 0.0
        average_loss = 0.0
        for local_step in trange(
            self.config.diffusion_train_steps,
            desc=f"CFDG refresh at online step {online_step}",
            leave=False,
        ):
            clean, labels = self._sample_training_vectors(
                offline_dataset,
                online_buffer,
                self.config.diffusion_batch_size,
            )
            dropped = torch.rand(
                labels.shape[0], device=self.device
            ) < self.config.classifier_free_drop_probability
            training_labels = labels.clone()
            training_labels[dropped] = self.NULL_LABEL

            sigma = torch.exp(
                self.config.diffusion_p_mean
                + self.config.diffusion_p_std
                * torch.randn(clean.shape[0], device=self.device)
            )
            noise = torch.randn_like(clean)
            noisy = clean + sigma.unsqueeze(-1) * noise
            denoised = self.model(noisy, sigma, training_labels)
            sigma_data = self.config.diffusion_sigma_data
            weights = (
                sigma.square() + sigma_data**2
            ) / ((sigma * sigma_data).square())
            loss = (weights.unsqueeze(-1) * (denoised - clean).square()).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            self._update_ema()

            final_loss = float(loss.detach().item())
            average_loss += final_loss
            self.train_updates += 1
            if writer is not None and local_step % 1000 == 0:
                writer.add_scalar(
                    "diffusion/loss",
                    final_loss,
                    self.train_updates,
                )
                writer.add_scalar(
                    "diffusion/learning_rate",
                    scheduler.get_last_lr()[0],
                    self.train_updates,
                )

        return {
            "diffusion_loss": final_loss,
            "diffusion_average_loss": average_loss
            / max(1, self.config.diffusion_train_steps),
        }

    def _sigma_schedule(self) -> torch.Tensor:
        steps = self.config.diffusion_sampling_steps
        ramp = torch.linspace(0.0, 1.0, steps, device=self.device)
        minimum = self.config.edm_sigma_min ** (1.0 / self.config.edm_rho)
        maximum = self.config.edm_sigma_max ** (1.0 / self.config.edm_rho)
        sigmas = (maximum + ramp * (minimum - maximum)).pow(
            self.config.edm_rho
        )
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    @torch.no_grad()
    def _guided_denoise(
        self,
        samples: torch.Tensor,
        sigma_value: torch.Tensor,
        label: int,
    ) -> torch.Tensor:
        batch_size = samples.shape[0]
        sigma = torch.full(
            (batch_size,),
            float(sigma_value.item()),
            dtype=samples.dtype,
            device=self.device,
        )
        conditional_labels = torch.full(
            (batch_size,), label, dtype=torch.long, device=self.device
        )
        null_labels = torch.full(
            (batch_size,), self.NULL_LABEL, dtype=torch.long, device=self.device
        )
        conditional = self.ema_model(samples, sigma, conditional_labels)
        unconditional = self.ema_model(samples, sigma, null_labels)
        weight = self.config.classifier_free_guidance_weight
        return (1.0 + weight) * conditional - weight * unconditional

    @torch.no_grad()
    def sample_vectors(self, count: int, label: int) -> torch.Tensor:
        if count < 1:
            return torch.empty(
                (0, self.codec.transition_dim), device=self.device
            )
        self.ema_model.eval()
        sigmas = self._sigma_schedule()
        outputs = []
        for start in trange(
            0,
            count,
            self.config.generation_batch_size,
            desc=("Generate online synthetic" if label == 1 else "Generate offline synthetic"),
            leave=False,
        ):
            batch_size = min(
                self.config.generation_batch_size, count - start
            )
            samples = (
                torch.randn(
                    batch_size,
                    self.codec.transition_dim,
                    device=self.device,
                )
                * sigmas[0]
            )
            for index in range(len(sigmas) - 1):
                sigma_current = sigmas[index]
                sigma_next = sigmas[index + 1]
                gamma = 0.0
                if (
                    self.config.edm_s_min
                    <= float(sigma_current.item())
                    <= self.config.edm_s_max
                ):
                    gamma = min(
                        self.config.edm_s_churn
                        / self.config.diffusion_sampling_steps,
                        math.sqrt(2.0) - 1.0,
                    )
                sigma_hat = sigma_current * (1.0 + gamma)
                if gamma > 0.0:
                    epsilon = torch.randn_like(samples)
                    samples = samples + (
                        torch.sqrt(
                            (sigma_hat.square() - sigma_current.square()).clamp_min(0.0)
                        )
                        * self.config.edm_s_noise
                        * epsilon
                    )

                denoised = self._guided_denoise(samples, sigma_hat, label)
                derivative = (samples - denoised) / sigma_hat.clamp_min(1e-8)
                proposed = samples + (sigma_next - sigma_hat) * derivative

                if float(sigma_next.item()) != 0.0:
                    next_denoised = self._guided_denoise(
                        proposed, sigma_next, label
                    )
                    next_derivative = (
                        proposed - next_denoised
                    ) / sigma_next.clamp_min(1e-8)
                    samples = samples + (
                        sigma_next - sigma_hat
                    ) * (0.5 * derivative + 0.5 * next_derivative)
                else:
                    samples = proposed
            outputs.append(samples.cpu())
        return torch.cat(outputs, dim=0).to(self.device)

    def generate_to_buffers(
        self,
        offline_synthetic_buffer: ReplayBuffer,
        online_synthetic_buffer: ReplayBuffer,
        total_count: int,
    ) -> Dict[str, float]:
        online_count = int(round(total_count * self.config.online_synthetic_fraction))
        offline_count = total_count - online_count
        offline_vectors = self.sample_vectors(
            offline_count, self.OFFLINE_LABEL
        )
        online_vectors = self.sample_vectors(
            online_count, self.ONLINE_LABEL
        )
        offline_synthetic_buffer.add_batch(
            self.codec.unpack_tensor(offline_vectors)
        )
        online_synthetic_buffer.add_batch(
            self.codec.unpack_tensor(online_vectors)
        )
        return {
            "generated_offline": float(offline_count),
            "generated_online": float(online_count),
        }


# =============================================================================
# Batch construction and evaluation
# =============================================================================


def _concatenate_batches(*batches: NumpyBatch) -> NumpyBatch:
    nonempty = [batch for batch in batches if len(batch["rewards"]) > 0]
    if not nonempty:
        raise ValueError("No non-empty batches to concatenate.")
    return {
        key: np.concatenate([batch[key] for batch in nonempty], axis=0)
        for key in nonempty[0]
    }


def _empty_batch(observation_dim: int, action_dim: int) -> NumpyBatch:
    return {
        "observations": np.empty((0, observation_dim), dtype=np.float32),
        "actions": np.empty((0, action_dim), dtype=np.float32),
        "rewards": np.empty((0,), dtype=np.float32),
        "masks": np.empty((0,), dtype=np.float32),
        "next_observations": np.empty(
            (0, observation_dim), dtype=np.float32
        ),
    }


def _split_counts(total: int, proportions: Sequence[float]) -> Sequence[int]:
    raw = np.asarray(proportions, dtype=np.float64)
    raw = raw / raw.sum() * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for index in order[:remainder]:
            counts[index] += 1
    return counts.tolist()


def sample_cfdg_iql_batch(
    offline_dataset: ArrayDataset,
    online_buffer: ReplayBuffer,
    offline_synthetic_buffer: ReplayBuffer,
    online_synthetic_buffer: ReplayBuffer,
    config: TrainConfig,
    observation_dim: int,
    action_dim: int,
) -> NumpyBatch:
    synthetic_available = (
        offline_synthetic_buffer.size > 0
        and online_synthetic_buffer.size > 0
    )
    if not synthetic_available:
        offline_count, online_count = _split_counts(
            config.batch_size, [0.5, 0.5]
        )
        return _concatenate_batches(
            offline_dataset.sample(offline_count),
            online_buffer.sample(online_count),
        )

    synthetic_count = int(round(config.batch_size * config.synthetic_batch_ratio))
    real_count = config.batch_size - synthetic_count
    offline_real_count, online_real_count = _split_counts(
        real_count, [0.5, 0.5]
    )
    offline_syn_count, online_syn_count = _split_counts(
        synthetic_count,
        [1.0 - config.online_synthetic_fraction, config.online_synthetic_fraction],
    )

    batches = [
        offline_dataset.sample(offline_real_count)
        if offline_real_count > 0
        else _empty_batch(observation_dim, action_dim),
        online_buffer.sample(online_real_count)
        if online_real_count > 0
        else _empty_batch(observation_dim, action_dim),
        offline_synthetic_buffer.sample(offline_syn_count)
        if offline_syn_count > 0
        else _empty_batch(observation_dim, action_dim),
        online_synthetic_buffer.sample(online_syn_count)
        if online_syn_count > 0
        else _empty_batch(observation_dim, action_dim),
    ]
    combined = _concatenate_batches(*batches)
    permutation = np.random.permutation(config.batch_size)
    return {key: value[permutation] for key, value in combined.items()}


def _to_torch_batch(batch: NumpyBatch, device: torch.device) -> TorchBatch:
    return {
        "observations": torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=device
        ),
        "actions": torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=device
        ),
        "rewards": torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=device
        ).reshape(-1),
        "masks": torch.as_tensor(
            batch["masks"], dtype=torch.float32, device=device
        ).reshape(-1),
        "next_observations": torch.as_tensor(
            batch["next_observations"], dtype=torch.float32, device=device
        ),
    }


@torch.no_grad()
def evaluate(
    actor: GaussianPolicy,
    env: gym.Env,
    device: torch.device,
    episodes: int,
) -> Dict[str, float]:
    actor.eval()
    returns = []
    lengths = []
    for _ in range(episodes):
        observation = _reset_env(env)
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = actor.act(observation, device, deterministic=True)
            observation, reward, done, _ = _step_env(env, action)
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)
    actor.train()
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
    }


def _normalized_score(env: gym.Env, raw_return: float) -> Optional[float]:
    candidate = env
    for _ in range(16):
        if hasattr(candidate, "get_normalized_score"):
            try:
                return float(candidate.get_normalized_score(raw_return) * 100.0)
            except Exception:
                return None
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    return None


# =============================================================================
# Training
# =============================================================================


def _set_seed(seed: int, deterministic_torch: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _apply_domain_defaults(config: TrainConfig) -> None:
    name = config.env_name.lower()
    if config.expectile is None:
        config.expectile = 0.9 if "antmaze" in name else 0.7
    if config.advantage_temperature is None:
        # Common IQL settings used in the public PyTorch IQL/PEX implementation.
        config.advantage_temperature = 10.0 if "antmaze" in name else 3.0
    if config.synthetic_samples_per_refresh is None:
        number_of_refreshes = max(
            1, config.num_total_steps // config.diffusion_refresh_every
        )
        config.synthetic_samples_per_refresh = int(
            math.ceil(config.synthetic_buffer_size / number_of_refreshes)
        )


def _validate_config(config: TrainConfig) -> None:
    if config.batch_size < 3:
        raise ValueError("batch_size must be at least 3.")
    if config.diffusion_batch_size < 2:
        raise ValueError("diffusion_batch_size must be at least 2.")
    if config.diffusion_sampling_steps < 2:
        raise ValueError("diffusion_sampling_steps must be at least 2.")
    if not 0.0 < config.synthetic_batch_ratio < 1.0:
        raise ValueError("synthetic_batch_ratio must lie in (0,1).")
    if not 0.0 <= config.online_synthetic_fraction <= 1.0:
        raise ValueError("online_synthetic_fraction must lie in [0,1].")
    if not 0.0 < config.diffusion_online_fraction < 1.0:
        raise ValueError("diffusion_online_fraction must lie in (0,1).")
    if not 0.0 <= config.classifier_free_drop_probability < 1.0:
        raise ValueError(
            "classifier_free_drop_probability must lie in [0,1)."
        )
    if config.diffusion_refresh_every < 1:
        raise ValueError("diffusion_refresh_every must be positive.")
    if config.start_training < 0:
        raise ValueError("start_training cannot be negative.")


def train(config: TrainConfig) -> None:
    _apply_domain_defaults(config)
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)
    device = torch.device(config.device)

    train_env = wrap_gym(gym.make(config.env_name))
    eval_env = wrap_gym(gym.make(config.env_name))
    _seed_env(train_env, config.train_seed)
    _seed_env(eval_env, config.train_seed + config.eval_seed_offset)

    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise TypeError("CFDG requires a flat Box observation space.")
    if not isinstance(train_env.action_space, gym.spaces.Box):
        raise TypeError("CFDG requires a Box action space.")

    observation_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))
    offline_dataset = D4RLDataset(
        train_env, config.env_name, seed=config.train_seed
    )

    learner = IQLLearner(
        observation_dim,
        action_dim,
        train_env.action_space.low,
        train_env.action_space.high,
        config,
        device,
    )
    online_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        config.online_replay_size,
        seed=config.train_seed + 1,
    )
    offline_synthetic_capacity = max(
        1,
        int(round(config.synthetic_buffer_size * (1.0 - config.online_synthetic_fraction))),
    )
    online_synthetic_capacity = max(
        1, config.synthetic_buffer_size - offline_synthetic_capacity
    )
    offline_synthetic_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        offline_synthetic_capacity,
        seed=config.train_seed + 2,
    )
    online_synthetic_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        online_synthetic_capacity,
        seed=config.train_seed + 3,
    )

    codec = TransitionCodec(
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_low=train_env.action_space.low,
        action_high=train_env.action_space.high,
        normalize=config.normalize_diffusion_transitions,
        eps=config.diffusion_normalization_eps,
        mask_threshold=config.generated_mask_threshold,
    )
    codec.fit(offline_dataset)
    cfdg = CFDGModel(codec, config, device)

    run_name = f"{config.env_name}_seed{config.train_seed}_IQL_CFDG"
    log_dir = os.path.join(
        config.log_root, config.env_name.split("-")[0], run_name
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"env={config.env_name}\n"
        f"device={device}\n"
        f"base=IQL, expectile={config.expectile}, "
        f"adv_temperature={config.advantage_temperature}\n"
        f"offline_pretrain_steps={config.pretrain_steps}\n"
        f"online_steps={config.num_total_steps}\n"
        f"CFDG refresh={config.diffusion_refresh_every}, "
        f"diffusion_train_steps={config.diffusion_train_steps}\n"
        f"synthetic real ratios=offline:online:synthetic=1:1:1, "
        f"online-syn:offline-syn=8:2\n"
        f"synthetic_samples_per_refresh={config.synthetic_samples_per_refresh}\n"
        f"log_dir={log_dir}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Phase 1: offline IQL pre-training
    # -------------------------------------------------------------------------
    for pretrain_step in trange(
        config.pretrain_steps, desc="IQL offline pretraining"
    ):
        batch = _to_torch_batch(
            offline_dataset.sample(config.batch_size), device
        )
        metrics = learner.update(batch, offline_phase=True)
        if pretrain_step % config.log_every == 0:
            for key, value in metrics.items():
                writer.add_scalar(
                    f"offline-training/{key}", value, pretrain_step
                )
        if pretrain_step % config.eval_every == 0:
            result = evaluate(
                learner.actor, eval_env, device, config.eval_episodes
            )
            writer.add_scalar(
                "offline-evaluation/return", result["return"], pretrain_step
            )
            normalized = _normalized_score(eval_env, result["return"])
            if normalized is not None:
                writer.add_scalar(
                    "offline-evaluation/d4rl_normalized_score",
                    normalized,
                    pretrain_step,
                )
            writer.flush()

    learner.start_online_phase()

    # -------------------------------------------------------------------------
    # Phase 2: online fine-tuning with periodic CFDG augmentation
    # -------------------------------------------------------------------------
    observation = _reset_env(train_env)
    episode_return = 0.0
    episode_length = 0

    for online_step in trange(
        1, config.num_total_steps + 1, desc="IQL + CFDG online fine-tuning"
    ):
        action = learner.actor.act(
            observation, device, deterministic=False
        )
        next_observation, reward, done, info = _step_env(train_env, action)
        timeout = bool(info.get("TimeLimit.truncated", False))
        mask = 1.0 if (not done or timeout) else 0.0
        replay_reward = float(_transform_reward(reward, config.env_name))
        online_buffer.insert(
            observation,
            action,
            replay_reward,
            mask,
            next_observation,
        )

        episode_return += reward
        episode_length += 1
        observation = next_observation
        if done:
            writer.add_scalar(
                "online/episode_return", episode_return, online_step
            )
            writer.add_scalar(
                "online/episode_length", episode_length, online_step
            )
            observation = _reset_env(train_env)
            episode_return = 0.0
            episode_length = 0

        if (
            online_step % config.diffusion_refresh_every == 0
            and online_buffer.size > 0
        ):
            diffusion_metrics = cfdg.train_refresh(
                offline_dataset,
                online_buffer,
                writer,
                online_step,
            )
            generation_metrics = cfdg.generate_to_buffers(
                offline_synthetic_buffer,
                online_synthetic_buffer,
                int(config.synthetic_samples_per_refresh),
            )
            for key, value in {
                **diffusion_metrics,
                **generation_metrics,
                "offline_synthetic_buffer_size": float(
                    offline_synthetic_buffer.size
                ),
                "online_synthetic_buffer_size": float(
                    online_synthetic_buffer.size
                ),
            }.items():
                writer.add_scalar(f"cfdg/{key}", value, online_step)
            writer.flush()

        metrics = {}
        if online_buffer.size > config.start_training:
            for _ in range(config.online_updates_per_step):
                numpy_batch = sample_cfdg_iql_batch(
                    offline_dataset,
                    online_buffer,
                    offline_synthetic_buffer,
                    online_synthetic_buffer,
                    config,
                    observation_dim,
                    action_dim,
                )
                metrics = learner.update(
                    _to_torch_batch(numpy_batch, device),
                    offline_phase=False,
                )

        if metrics and online_step % config.log_every == 0:
            for key, value in metrics.items():
                writer.add_scalar(
                    f"online-training/{key}", value, online_step
                )
            writer.add_scalar(
                "buffers/online", online_buffer.size, online_step
            )
            writer.add_scalar(
                "buffers/offline_synthetic",
                offline_synthetic_buffer.size,
                online_step,
            )
            writer.add_scalar(
                "buffers/online_synthetic",
                online_synthetic_buffer.size,
                online_step,
            )

        if online_step % config.eval_every == 0:
            result = evaluate(
                learner.actor, eval_env, device, config.eval_episodes
            )
            writer.add_scalar(
                "evaluation/return", result["return"], online_step
            )
            writer.add_scalar(
                "evaluation/length", result["length"], online_step
            )
            normalized = _normalized_score(eval_env, result["return"])
            message = (
                f"[eval] step={online_step}, "
                f"return={result['return']:.3f}, "
                f"length={result['length']:.1f}"
            )
            if normalized is not None:
                writer.add_scalar(
                    "evaluation/d4rl_normalized_score",
                    normalized,
                    online_step,
                )
                message += f", normalized_score={normalized:.3f}"
            print(message, flush=True)
            writer.flush()

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "actor": learner.actor.state_dict(),
                "critic": learner.critic.state_dict(),
                "target_critic": learner.target_critic.state_dict(),
                "value": learner.value.state_dict(),
                "diffusion": cfdg.model.state_dict(),
                "diffusion_ema": cfdg.ema_model.state_dict(),
                "diffusion_optimizer": cfdg.optimizer.state_dict(),
                "transition_mean": codec.mean,
                "transition_std": codec.std,
                "config": asdict(config),
            },
            os.path.join(checkpoint_dir, "final.pt"),
        )

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train(pyrallis.parse(config_class=TrainConfig))
