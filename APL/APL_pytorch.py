"""Single-file PyTorch implementation of Adaptive Policy Learning (APL).

This file implements the two concrete algorithms defined in:

    Adaptive Policy Learning for Offline-to-Online Reinforcement Learning
    (AAAI 2023).

The paper presents APL as a framework and instantiates it in two ways:

1. GCQL (Greedy-Conservative Q-ensemble Learning)
   - REDQ/SAC is the optimistic online update.
   - A CQL conservative critic penalty is applied only when the entire update
     batch is drawn from the offline/growing replay buffer.
   - The conservative penalty is disabled for batches drawn from the small
     near-on-policy online FIFO buffer.

2. GCTD3BC (Greedy-Conservative TD3+BC)
   - TD3 is the optimistic online update.
   - The behavior-cloning policy penalty is applied only for batches drawn
     from the offline/growing replay buffer.
   - Online-buffer batches use the greedy TD3 actor objective without BC.

Both variants use the paper's Online-Offline Replay Buffer (OORB):

- ``offline_buffer`` starts with the complete D4RL dataset and additionally
  receives every transition collected online.
- ``online_buffer`` is a small FIFO buffer containing only recent online data.
- Each gradient update chooses one complete batch from the online buffer with
  probability ``p`` after ``T_s`` online transitions have been collected;
  otherwise it chooses one complete batch from the offline/growing buffer.
- The source of the whole batch determines W(s,a): online -> 0, offline -> 1.

The training schedule follows Algorithm 1 and the experimental section:

- 100,000 initial offline gradient updates.
- Repeated blocks of 1,000 online interactions and 10,000 gradient updates.
- 100,000 total online interactions.
- online-buffer capacity 20,000 and offline-buffer capacity 3,000,000.
- GCQL uses p=0.5 and five critics; GCTD3BC uses p=0.1.

No author-released APL repository is linked by the paper, the AAAI page, the
Microsoft Research page, or public paper-code indexes at the time this file was
prepared. Therefore, this is a paper-faithful implementation rather than a
claim of line-by-line migration from unavailable official source code. Details
that the APL paper delegates to its base algorithms use standard REDQ, CQL,
and TD3+BC formulations and are exposed as explicit configuration options.

The default configuration runs GCQL on ``hopper-medium-replay-v2``. The paper
used the earlier D4RL v0 task identifiers; v2 is used here because it is the
commonly available equivalent in current D4RL installations.
"""

import math
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import d4rl  # noqa: F401: importing D4RL registers Gym environments
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


@dataclass
class TrainConfig:
    # APL concrete implementation: GCQL | GCTD3BC.
    algorithm: str = "GCQL"
    env_name: str = "hopper-medium-replay-v2"

    # Network and optimizer settings delegated by APL to the base algorithms.
    hidden_dim: int = 256
    hidden_layers: int = 2
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 5e-3
    batch_size: int = 256

    # Paper-level APL/OORB schedule.
    initial_offline_updates: int = 100_000  # T_initial
    interaction_steps_per_iteration: int = 1_000  # T_on
    updates_per_iteration: int = 10_000  # T_off
    total_online_steps: int = 100_000  # S_T
    online_sampling_start: int = 10_000  # T_s
    online_buffer_size: int = 20_000
    offline_buffer_size: int = 3_000_000
    online_sampling_probability: Optional[float] = None

    # GCQL: REDQ/SAC + source-gated CQL.
    num_critics: int = 5
    num_min_qs: int = 2
    init_temperature: float = 0.2
    target_entropy: Optional[float] = None
    backup_entropy: bool = True
    actor_log_std_min: float = -20.0
    actor_log_std_max: float = 2.0
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0
    cql_num_actions: int = 10
    # Eq. (5) samples current-policy actions. Turning this on recovers the
    # density-corrected CQL(H) engineering variant, but it is off by default so
    # the objective follows the APL paper's displayed equation.
    cql_importance_sample: bool = False

    # GCTD3BC: TD3 + source-gated behavior cloning.
    td3_bc_alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    exploration_noise: float = 0.1

    # Original TD3+BC normalizes states. GCQL normally uses raw D4RL states.
    # None selects False for GCQL and True for GCTD3BC.
    normalize_states: Optional[bool] = None
    state_normalization_eps: float = 1e-3

    eval_episodes: int = 10
    eval_every_iterations: int = 1
    log_every_updates: int = 1_000

    train_seed: int = 42
    eval_seed_offset: int = 42
    deterministic_torch: bool = False

    checkpoints_path: Optional[str] = None
    log_root: str = "logs/APL_PyTorch"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def _resolve_config(config: TrainConfig) -> None:
    algorithm = config.algorithm.upper()
    if algorithm not in {"GCQL", "GCTD3BC"}:
        raise ValueError("algorithm must be either 'GCQL' or 'GCTD3BC'.")
    config.algorithm = algorithm

    if config.online_sampling_probability is None:
        # Experimental section: p=0.5 for GCQL and p=0.1 for GCTD3BC.
        config.online_sampling_probability = 0.5 if algorithm == "GCQL" else 0.1

    if config.normalize_states is None:
        config.normalize_states = algorithm == "GCTD3BC"


def _validate_config(config: TrainConfig) -> None:
    _resolve_config(config)

    positive_ints = {
        "hidden_dim": config.hidden_dim,
        "hidden_layers": config.hidden_layers,
        "batch_size": config.batch_size,
        "initial_offline_updates": config.initial_offline_updates,
        "interaction_steps_per_iteration": config.interaction_steps_per_iteration,
        "updates_per_iteration": config.updates_per_iteration,
        "total_online_steps": config.total_online_steps,
        "online_buffer_size": config.online_buffer_size,
        "offline_buffer_size": config.offline_buffer_size,
        "eval_episodes": config.eval_episodes,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"{name} must be positive.")

    if not 0.0 <= float(config.online_sampling_probability) <= 1.0:
        raise ValueError("online_sampling_probability must be in [0, 1].")
    if config.online_sampling_start < 0:
        raise ValueError("online_sampling_start cannot be negative.")
    if config.offline_buffer_size <= config.online_buffer_size:
        raise ValueError(
            "offline_buffer_size should exceed online_buffer_size for the "
            "paper's two-level OORB design."
        )
    if config.algorithm == "GCQL":
        if config.num_critics < 2:
            raise ValueError("GCQL requires at least two critics.")
        if not 1 <= config.num_min_qs <= config.num_critics:
            raise ValueError("num_min_qs must be in [1, num_critics].")
        if config.cql_num_actions < 1:
            raise ValueError("cql_num_actions must be positive.")
    else:
        if config.policy_frequency < 1:
            raise ValueError("policy_frequency must be positive.")


# -----------------------------------------------------------------------------
# Environment compatibility
# -----------------------------------------------------------------------------


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


class ObservationNormalizer(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        return (observation - self.mean) / self.std


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
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


# -----------------------------------------------------------------------------
# Dataset and OORB
# -----------------------------------------------------------------------------


def _load_d4rl_dataset(env: gym.Env) -> NumpyBatch:
    raw = d4rl.qlearning_dataset(env)
    observations = np.asarray(raw["observations"], dtype=np.float32)
    next_observations = np.asarray(raw["next_observations"], dtype=np.float32)
    actions = np.clip(
        np.asarray(raw["actions"], dtype=np.float32), -1.0 + 1e-5, 1.0 - 1e-5
    )
    rewards = np.asarray(raw["rewards"], dtype=np.float32).reshape(-1)
    terminals = np.asarray(raw["terminals"], dtype=np.float32).reshape(-1)

    # Time-limit truncations remain bootstrappable, matching common D4RL use.
    masks = 1.0 - terminals
    dones = terminals.astype(bool)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "masks": masks.astype(np.float32),
        "dones": dones,
        "next_observations": next_observations,
    }


def _compute_state_statistics(
    observations: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = observations.mean(axis=0, keepdims=False).astype(np.float32)
    std = (observations.std(axis=0, keepdims=False) + eps).astype(np.float32)
    return mean, std


def _normalize_dataset_states(
    dataset: NumpyBatch,
    mean: np.ndarray,
    std: np.ndarray,
) -> NumpyBatch:
    output = dict(dataset)
    output["observations"] = (
        (dataset["observations"] - mean) / std
    ).astype(np.float32)
    output["next_observations"] = (
        (dataset["next_observations"] - mean) / std
    ).astype(np.float32)
    return output


class ReplayBuffer:
    """Fixed-capacity FIFO ring buffer used for both OORB levels."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        seed: Optional[int] = None,
    ):
        if capacity < 1:
            raise ValueError("Replay-buffer capacity must be positive.")
        self.capacity = int(capacity)
        self.size = 0
        self.pointer = 0
        self.observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=bool)
        self.next_observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.rng = np.random.default_rng(seed)

    def load_dataset(self, dataset: Mapping[str, np.ndarray]) -> None:
        dataset_size = len(dataset["observations"])
        if dataset_size > self.capacity:
            raise ValueError(
                f"Dataset size {dataset_size} exceeds capacity {self.capacity}."
            )
        self.observations[:dataset_size] = dataset["observations"]
        self.actions[:dataset_size] = dataset["actions"]
        self.rewards[:dataset_size] = dataset["rewards"]
        self.masks[:dataset_size] = dataset["masks"]
        self.dones[:dataset_size] = dataset["dones"]
        self.next_observations[:dataset_size] = dataset["next_observations"]
        self.size = dataset_size
        self.pointer = dataset_size % self.capacity

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        index = self.pointer
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.masks[index] = mask
        self.dones[index] = done
        self.next_observations[index] = next_observation
        self.pointer = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size < 1:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "dones": self.dones[indices],
            "next_observations": self.next_observations[indices],
        }


class OnlineOfflineReplayBuffer:
    """Paper-defined Online-Offline Replay Buffer (OORB)."""

    def __init__(
        self,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        online_probability: float,
        online_sampling_start: int,
        seed: int,
    ):
        self.offline_buffer = offline_buffer
        self.online_buffer = online_buffer
        self.online_probability = float(online_probability)
        self.online_sampling_start = int(online_sampling_start)
        self.rng = np.random.default_rng(seed)

    def insert_online_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        transition = (
            observation,
            action,
            reward,
            mask,
            done,
            next_observation,
        )
        # APL stores every online transition in both OORB levels.
        self.online_buffer.insert(*transition)
        self.offline_buffer.insert(*transition)

    def sample(
        self,
        batch_size: int,
        total_online_steps: int,
    ) -> Tuple[NumpyBatch, bool]:
        can_use_online = (
            total_online_steps > self.online_sampling_start
            and self.online_buffer.size > 0
        )
        use_online = can_use_online and (
            self.rng.random() < self.online_probability
        )
        if use_online:
            return self.online_buffer.sample(batch_size), False
        return self.offline_buffer.sample(batch_size), True


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _init_linear(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    output_gain: float = 1.0,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        linear = nn.Linear(current_dim, hidden_dim)
        _init_linear(linear, gain=math.sqrt(2.0))
        layers.extend([linear, nn.ReLU()])
        current_dim = hidden_dim
    output = nn.Linear(current_dim, output_dim)
    _init_linear(output, gain=output_gain)
    layers.append(output)
    return nn.Sequential(*layers)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


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
        "dones": torch.as_tensor(
            batch["dones"], dtype=torch.bool, device=device
        ).reshape(-1),
        "next_observations": torch.as_tensor(
            batch["next_observations"], dtype=torch.float32, device=device
        ),
    }


# -----------------------------------------------------------------------------
# GCQL networks and learner
# -----------------------------------------------------------------------------


class GaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        trunk_layers = []
        current_dim = observation_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            _init_linear(linear, gain=math.sqrt(2.0))
            trunk_layers.extend([linear, nn.ReLU()])
            current_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.mean = nn.Linear(current_dim, action_dim)
        self.log_std = nn.Linear(current_dim, action_dim)
        _init_linear(self.mean, gain=1e-2)
        _init_linear(self.log_std, gain=1e-2)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def distribution(self, observations: torch.Tensor) -> Normal:
        features = self.trunk(observations)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            self.log_std_min, self.log_std_max
        )
        return Normal(mean, log_std.exp())

    @staticmethod
    def _log_prob_from_pre_tanh(
        distribution: Normal,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        base_log_prob = distribution.log_prob(pre_tanh)
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (base_log_prob - correction).sum(dim=-1)

    def sample(
        self,
        observations: torch.Tensor,
        reparameterize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        pre_tanh = (
            distribution.rsample() if reparameterize else distribution.sample()
        )
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(distribution, pre_tanh)
        return action, log_prob

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observations).mean)

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
        if deterministic:
            action = self.mode(tensor)
        else:
            action, _ = self.sample(tensor, reparameterize=False)
        return action.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.network = _build_mlp(
            observation_dim + action_dim,
            hidden_dims,
            1,
            output_gain=1.0,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1)).squeeze(-1)


class CriticEnsemble(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        ensemble_size: int,
    ):
        super().__init__()
        self.critics = nn.ModuleList(
            [
                QNetwork(observation_dim, action_dim, hidden_dims)
                for _ in range(ensemble_size)
            ]
        )

    @property
    def ensemble_size(self) -> int:
        return len(self.critics)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if indices is None:
            selected = self.critics
        else:
            selected = [self.critics[int(index)] for index in indices.tolist()]
        return torch.stack(
            [critic(observations, actions) for critic in selected], dim=0
        )


class GCQLLearner:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        config: TrainConfig,
        device: torch.device,
    ):
        self.actor = GaussianActor(
            observation_dim,
            action_dim,
            hidden_dims,
            config.actor_log_std_min,
            config.actor_log_std_max,
        ).to(device)
        self.critic = CriticEnsemble(
            observation_dim,
            action_dim,
            hidden_dims,
            config.num_critics,
        ).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

        self.log_temperature = torch.tensor(
            math.log(config.init_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=config.temperature_learning_rate
        )

        self.target_entropy = (
            -float(action_dim)
            if config.target_entropy is None
            else float(config.target_entropy)
        )
        self.discount = float(config.discount)
        self.tau = float(config.tau)
        self.num_min_qs = int(config.num_min_qs)
        self.backup_entropy = bool(config.backup_entropy)
        self.cql_alpha = float(config.cql_alpha)
        self.cql_temperature = float(config.cql_temperature)
        self.cql_num_actions = int(config.cql_num_actions)
        self.cql_importance_sample = bool(config.cql_importance_sample)
        self.action_dim = int(action_dim)
        self.device = device

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _sample_target_indices(self) -> torch.Tensor:
        return torch.randperm(
            self.critic.ensemble_size, device=self.device
        )[: self.num_min_qs]

    def _cql_penalty(self, batch: TorchBatch) -> torch.Tensor:
        """Eq. (5): logsumexp over current-policy actions minus data Q."""

        observations = batch["observations"]
        batch_size = observations.shape[0]
        repeated_observations = observations.unsqueeze(1).expand(
            batch_size, self.cql_num_actions, observations.shape[-1]
        )
        flat_observations = repeated_observations.reshape(
            batch_size * self.cql_num_actions, -1
        )

        with torch.no_grad():
            sampled_actions, sampled_log_prob = self.actor.sample(
                flat_observations, reparameterize=False
            )

        sampled_q = self.critic(flat_observations, sampled_actions)
        sampled_q = sampled_q.reshape(
            self.critic.ensemble_size, batch_size, self.cql_num_actions
        )

        if self.cql_importance_sample:
            sampled_log_prob = sampled_log_prob.reshape(
                1, batch_size, self.cql_num_actions
            )
            sampled_q = sampled_q - sampled_log_prob

        temperature = self.cql_temperature
        conservative_ood = temperature * torch.logsumexp(
            sampled_q / temperature, dim=-1
        )
        data_q = self.critic(batch["observations"], batch["actions"])
        return (conservative_ood - data_q).mean()

    def update_critic(
        self,
        batch: TorchBatch,
        source_is_offline: bool,
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                batch["next_observations"], reparameterize=False
            )
            target_indices = self._sample_target_indices()
            target_qs = self.target_critic(
                batch["next_observations"],
                next_actions,
                indices=target_indices,
            )
            next_q = target_qs.min(dim=0).values
            if self.backup_entropy:
                next_q = next_q - self.temperature.detach() * next_log_prob
            target_q = (
                batch["rewards"]
                + self.discount * batch["masks"] * next_q
            )

        predicted_qs = self.critic(batch["observations"], batch["actions"])
        bellman_loss = (
            predicted_qs - target_q.unsqueeze(0)
        ).pow(2).mean()

        if source_is_offline:
            cql_penalty = self._cql_penalty(batch)
        else:
            cql_penalty = torch.zeros((), device=self.device)

        critic_loss = bellman_loss + self.cql_alpha * cql_penalty

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        _soft_update(self.target_critic, self.critic, self.tau)

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "bellman_loss": float(bellman_loss.detach().item()),
            "cql_penalty": float(cql_penalty.detach().item()),
            "q": float(predicted_qs.detach().mean().item()),
            "target_q": float(target_q.detach().mean().item()),
        }

    def update_actor_and_temperature(
        self,
        batch: TorchBatch,
    ) -> Dict[str, float]:
        actions, log_prob = self.actor.sample(
            batch["observations"], reparameterize=True
        )

        # Freeze critic parameters while preserving dQ/da for the actor.
        critic_requires_grad = [
            parameter.requires_grad for parameter in self.critic.parameters()
        ]
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        q_values = self.critic(batch["observations"], actions).mean(dim=0)
        actor_loss = (
            self.temperature.detach() * log_prob - q_values
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter, requires_grad in zip(
            self.critic.parameters(), critic_requires_grad
        ):
            parameter.requires_grad_(requires_grad)

        temperature_loss = -(
            self.log_temperature
            * (log_prob.detach() + self.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "entropy": float((-log_prob).detach().mean().item()),
            "temperature": float(self.temperature.detach().item()),
            "temperature_loss": float(temperature_loss.detach().item()),
        }

    def update(
        self,
        batch: TorchBatch,
        source_is_offline: bool,
    ) -> Dict[str, float]:
        critic_info = self.update_critic(batch, source_is_offline)
        actor_info = self.update_actor_and_temperature(batch)
        return {**critic_info, **actor_info}

    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        return self.actor.act(observation, self.device, deterministic)

    def checkpoint(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
        }


# -----------------------------------------------------------------------------
# GCTD3BC networks and learner
# -----------------------------------------------------------------------------


class DeterministicActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.network = _build_mlp(
            observation_dim,
            hidden_dims,
            action_dim,
            output_gain=1e-2,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observations))

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        device: torch.device,
        noise_std: float,
        deterministic: bool,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action = self(tensor).squeeze(0)
        if not deterministic and noise_std > 0.0:
            action = action + noise_std * torch.randn_like(action)
        return action.clamp(-1.0, 1.0).cpu().numpy()


class TwinCritic(nn.Module):
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

    def q1_only(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.q1(observations, actions)


class GCTD3BCLearner:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        config: TrainConfig,
        device: torch.device,
    ):
        self.actor = DeterministicActor(
            observation_dim, action_dim, hidden_dims
        ).to(device)
        self.target_actor = deepcopy(self.actor).to(device)
        self.target_actor.requires_grad_(False)

        self.critic = TwinCritic(
            observation_dim, action_dim, hidden_dims
        ).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

        self.discount = float(config.discount)
        self.tau = float(config.tau)
        self.policy_noise = float(config.policy_noise)
        self.noise_clip = float(config.noise_clip)
        self.policy_frequency = int(config.policy_frequency)
        self.exploration_noise = float(config.exploration_noise)
        self.td3_bc_alpha = float(config.td3_bc_alpha)
        self.total_updates = 0
        self.device = device

    def update_critic(self, batch: TorchBatch) -> Dict[str, float]:
        with torch.no_grad():
            noise = torch.randn_like(batch["actions"]) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (
                self.target_actor(batch["next_observations"]) + noise
            ).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_critic(
                batch["next_observations"], next_actions
            )
            target_q = (
                batch["rewards"]
                + self.discount
                * batch["masks"]
                * torch.minimum(target_q1, target_q2)
            )

        q1, q2 = self.critic(batch["observations"], batch["actions"])
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "q1": float(q1.detach().mean().item()),
            "q2": float(q2.detach().mean().item()),
            "target_q": float(target_q.detach().mean().item()),
        }

    def update_actor(
        self,
        batch: TorchBatch,
        source_is_offline: bool,
    ) -> Dict[str, float]:
        policy_actions = self.actor(batch["observations"])
        q = self.critic.q1_only(batch["observations"], policy_actions)

        # Original TD3+BC normalization of the value term. In Eq. (8), this is
        # lambda Q. W(s,a) only gates the BC term.
        value_scale = self.td3_bc_alpha / q.abs().mean().detach().clamp_min(1e-6)
        bc_loss = F.mse_loss(policy_actions, batch["actions"])
        source_weight = 1.0 if source_is_offline else 0.0
        actor_loss = -value_scale * q.mean() + source_weight * bc_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        _soft_update(self.target_actor, self.actor, self.tau)
        _soft_update(self.target_critic, self.critic, self.tau)

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "bc_loss": float(bc_loss.detach().item()),
            "value_scale": float(value_scale.detach().item()),
        }

    def update(
        self,
        batch: TorchBatch,
        source_is_offline: bool,
    ) -> Dict[str, float]:
        self.total_updates += 1
        critic_info = self.update_critic(batch)
        actor_info: Dict[str, float] = {}
        if self.total_updates % self.policy_frequency == 0:
            actor_info = self.update_actor(batch, source_is_offline)
        return {**critic_info, **actor_info}

    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        return self.actor.act(
            observation,
            self.device,
            noise_std=self.exploration_noise,
            deterministic=deterministic,
        )

    def checkpoint(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "total_updates": self.total_updates,
        }


# -----------------------------------------------------------------------------
# Evaluation and training
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    learner,
    env: gym.Env,
    num_episodes: int,
) -> Dict[str, float]:
    returns = []
    lengths = []
    for _ in range(num_episodes):
        observation = _reset_env(env)
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = learner.act(observation, deterministic=True)
            observation, reward, done, _ = _step_env(env, action)
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
    }


def _get_normalized_score(env: gym.Env, raw_return: float) -> Optional[float]:
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


def _write_metrics(
    writer: SummaryWriter,
    prefix: str,
    metrics: Mapping[str, float],
    step: int,
) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def _evaluate_and_log(
    learner,
    eval_env: gym.Env,
    writer: SummaryWriter,
    step: int,
    num_episodes: int,
    prefix: str,
) -> None:
    eval_info = evaluate(learner, eval_env, num_episodes)
    _write_metrics(writer, prefix, eval_info, step)
    normalized_score = _get_normalized_score(eval_env, eval_info["return"])
    message = (
        f"[{prefix}] step={step}, return={eval_info['return']:.3f}, "
        f"length={eval_info['length']:.1f}"
    )
    if normalized_score is not None:
        writer.add_scalar(f"{prefix}/d4rl_normalized_score", normalized_score, step)
        message += f", normalized_score={normalized_score:.3f}"
    print(message, flush=True)
    writer.flush()


def train(config: TrainConfig) -> None:
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)
    device = torch.device(config.device)

    # Load the D4RL dataset before adding an observation-normalization wrapper.
    dataset_env = wrap_gym(gym.make(config.env_name), rescale_actions=True)
    raw_dataset = _load_d4rl_dataset(dataset_env)

    state_mean = np.zeros(
        raw_dataset["observations"].shape[-1], dtype=np.float32
    )
    state_std = np.ones_like(state_mean)
    if config.normalize_states:
        state_mean, state_std = _compute_state_statistics(
            raw_dataset["observations"], config.state_normalization_eps
        )
        dataset = _normalize_dataset_states(raw_dataset, state_mean, state_std)
    else:
        dataset = raw_dataset

    train_env = wrap_gym(gym.make(config.env_name), rescale_actions=True)
    eval_env = wrap_gym(gym.make(config.env_name), rescale_actions=True)
    if config.normalize_states:
        train_env = ObservationNormalizer(train_env, state_mean, state_std)
        eval_env = ObservationNormalizer(eval_env, state_mean, state_std)

    _seed_env(train_env, config.train_seed)
    _seed_env(eval_env, config.train_seed + config.eval_seed_offset)

    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise TypeError("APL requires a flat Box observation space.")
    if not isinstance(train_env.action_space, gym.spaces.Box):
        raise TypeError("APL requires a continuous Box action space.")

    observation_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))
    hidden_dims = tuple(config.hidden_dim for _ in range(config.hidden_layers))

    offline_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        config.offline_buffer_size,
        seed=config.train_seed,
    )
    offline_buffer.load_dataset(dataset)
    online_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        config.online_buffer_size,
        seed=config.train_seed + 1,
    )
    oorb = OnlineOfflineReplayBuffer(
        offline_buffer=offline_buffer,
        online_buffer=online_buffer,
        online_probability=float(config.online_sampling_probability),
        online_sampling_start=config.online_sampling_start,
        seed=config.train_seed + 2,
    )

    if config.algorithm == "GCQL":
        learner = GCQLLearner(
            observation_dim,
            action_dim,
            hidden_dims,
            config,
            device,
        )
    else:
        learner = GCTD3BCLearner(
            observation_dim,
            action_dim,
            hidden_dims,
            config,
            device,
        )

    run_name = (
        f"{config.algorithm}_{config.env_name}_seed{config.train_seed}_"
        f"p{float(config.online_sampling_probability):.2f}"
    )
    log_dir = os.path.join(
        config.log_root,
        config.algorithm,
        config.env_name.split("-")[0],
        run_name,
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"algorithm={config.algorithm}\n"
        f"env={config.env_name}\n"
        f"device={device}\n"
        f"initial_offline_updates={config.initial_offline_updates}\n"
        f"T_on={config.interaction_steps_per_iteration}, "
        f"T_off={config.updates_per_iteration}, "
        f"S_T={config.total_online_steps}\n"
        f"OORB: p={float(config.online_sampling_probability):.3f}, "
        f"T_s={config.online_sampling_start}, "
        f"online_capacity={config.online_buffer_size}, "
        f"offline_capacity={config.offline_buffer_size}\n"
        f"normalize_states={config.normalize_states}\n"
        f"log_dir={log_dir}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Initial offline pre-training: W(s,a)=1 for every batch.
    # ------------------------------------------------------------------
    for update_step in trange(
        1,
        config.initial_offline_updates + 1,
        desc=f"{config.algorithm} offline pretraining",
    ):
        batch = offline_buffer.sample(config.batch_size)
        metrics = learner.update(
            _to_torch_batch(batch, device), source_is_offline=True
        )
        if update_step % config.log_every_updates == 0:
            _write_metrics(writer, "offline-training", metrics, update_step)

    _evaluate_and_log(
        learner,
        eval_env,
        writer,
        step=0,
        num_episodes=config.eval_episodes,
        prefix="evaluation",
    )

    # ------------------------------------------------------------------
    # Paper's block-interleaved online phase.
    # ------------------------------------------------------------------
    observation = _reset_env(train_env)
    episode_return = 0.0
    episode_length = 0
    total_online_steps = 0
    total_online_updates = 0
    iteration = 0

    while total_online_steps < config.total_online_steps:
        iteration += 1
        remaining = config.total_online_steps - total_online_steps
        collection_steps = min(
            config.interaction_steps_per_iteration, remaining
        )

        for _ in trange(
            collection_steps,
            desc=f"APL collect iteration {iteration}",
            leave=False,
        ):
            action = learner.act(observation, deterministic=False)
            next_observation, reward, done, info = _step_env(train_env, action)
            timeout = bool(info.get("TimeLimit.truncated", False))
            mask = 1.0 if (not done or timeout) else 0.0

            oorb.insert_online_transition(
                observation=observation,
                action=action,
                reward=reward,
                mask=mask,
                done=done,
                next_observation=next_observation,
            )

            episode_return += reward
            episode_length += 1
            total_online_steps += 1
            observation = next_observation

            if done:
                writer.add_scalar(
                    "interaction/episode_return", episode_return, total_online_steps
                )
                writer.add_scalar(
                    "interaction/episode_length", episode_length, total_online_steps
                )
                observation = _reset_env(train_env)
                episode_return = 0.0
                episode_length = 0

        online_batches = 0
        offline_batches = 0
        for _ in trange(
            config.updates_per_iteration,
            desc=f"APL update iteration {iteration}",
            leave=False,
        ):
            batch, source_is_offline = oorb.sample(
                config.batch_size, total_online_steps
            )
            if source_is_offline:
                offline_batches += 1
            else:
                online_batches += 1

            total_online_updates += 1
            metrics = learner.update(
                _to_torch_batch(batch, device),
                source_is_offline=source_is_offline,
            )
            if total_online_updates % config.log_every_updates == 0:
                _write_metrics(
                    writer,
                    "online-training",
                    metrics,
                    total_online_updates,
                )
                writer.add_scalar(
                    "online-training/source_is_offline",
                    float(source_is_offline),
                    total_online_updates,
                )

        total_batches = online_batches + offline_batches
        writer.add_scalar(
            "oorb/online_batch_fraction",
            online_batches / max(total_batches, 1),
            total_online_steps,
        )
        writer.add_scalar(
            "oorb/online_buffer_size",
            online_buffer.size,
            total_online_steps,
        )
        writer.add_scalar(
            "oorb/offline_buffer_size",
            offline_buffer.size,
            total_online_steps,
        )

        if iteration % config.eval_every_iterations == 0:
            _evaluate_and_log(
                learner,
                eval_env,
                writer,
                step=total_online_steps,
                num_episodes=config.eval_episodes,
                prefix="evaluation",
            )

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = learner.checkpoint()
        checkpoint.update(
            {
                "config": asdict(config),
                "state_mean": state_mean,
                "state_std": state_std,
                "total_online_steps": total_online_steps,
                "total_online_updates": total_online_updates,
            }
        )
        torch.save(checkpoint, os.path.join(checkpoint_dir, "final.pt"))

    writer.close()
    dataset_env.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train(pyrallis.parse(config_class=TrainConfig))
