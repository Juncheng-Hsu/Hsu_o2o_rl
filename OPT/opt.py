"""Single-file PyTorch implementation of OPT for offline-to-online RL.

Algorithmic source of truth:
  1. Shin et al., Online Pre-Training for Offline-to-Online Reinforcement Learning, ICML 2025.
  2. The authors' official PyTorch implementation: https://github.com/LGAI-Research/opt

This file keeps a single-file/TensorBoard interface, but the OPT learning rules follow the
paper and released code: TD3+BC or SPOT offline pre-training; N_tau frozen-policy online
collection; Hot-Plug differentiable online pre-training of a freshly initialized critic;
density-ratio balanced replay; and actor optimization using
(1-kappa) Q_off + kappa Q_on.

Environment handling:
  * Official-paper tasks use the paper/official-repository settings.
  * The project's additional RLPD tasks (MuJoCo medium-expert and Adroit human/expert)
    were not reported by OPT. They use a fixed, explicitly documented same-domain transfer
    rule rather than invented per-task tuning. At startup the script prints whether the
    selected environment is 'official' or 'adapted'.

Experiment-protocol controls such as offline_steps, online_steps, eval_every,
eval_episodes, seed and device remain command-line configurable.
"""

import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Tuple

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import d4rl  # noqa: F401: importing d4rl registers Gym environments
import gym
import numpy as np
import pyrallis
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


NumpyBatch = Dict[str, np.ndarray]
TorchBatch = Dict[str, torch.Tensor]
GOAL_ENVS = ("antmaze", "pen", "door", "hammer", "relocate")
LOCOMOTION_ENVS = ("halfcheetah", "hopper", "walker2d")


@dataclass
class TrainConfig:
    env_name: str = "hopper-medium-v2"

    # ------------------------------------------------------------------
    # Experiment protocol: intentionally user-configurable.
    # These are NOT silently changed by the environment preset.
    # ------------------------------------------------------------------
    offline_steps: int = 1_000_000
    online_steps: int = 1_000_000
    eval_every: int = 5_000
    eval_episodes: int = 10
    log_every: int = 1_000
    train_seed: int = 0
    eval_seed: int = 0
    deterministic_torch: bool = False
    checkpoints_path: Optional[str] = None
    log_root: str = "logs/OPT"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Environment-dependent algorithm parameters below are resolved
    # automatically from env_name. Set auto_env_config=False only for an
    # intentional ablation; formal baseline runs should leave it True.
    auto_env_config: bool = True
    strict_official_only: bool = False
    config_source: str = "unresolved"
    algo_type: str = "TD3"  # TD3 | SPOT

    # Shared TD3/SPOT settings from the official implementation.
    hidden_dim: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_layer_norm: bool = False
    critic_layer_norm: bool = False
    actor_init_w: Optional[float] = None
    critic_init_w: Optional[float] = None
    batch_size: int = 256
    replay_buffer_size: int = 2_000_000
    # Official PriorityReplayBuffer uses 1e6 capacity independently.
    priority_replay_buffer_size: int = 1_000_000
    discount: float = 0.99
    tau: float = 5e-3
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    normalize_states: bool = False
    normalize_rewards: bool = False

    # TD3+BC backbone.
    td3_bc_alpha: float = 2.5

    # SPOT backbone.
    vae_learning_rate: float = 1e-3
    vae_hidden_dim: int = 750
    vae_latent_dim: Optional[int] = None
    vae_beta: float = 0.5
    vae_steps: int = 100_000
    support_lambda: float = 1.0
    support_num_samples: int = 1
    support_use_iwae: bool = False
    support_lambda_cool: bool = False
    support_lambda_end: float = 0.2
    online_discount: float = 0.995

    # OPT-specific settings. N_tau=25k and N_pretrain=50k are fixed for all
    # environments in Appendix H of the paper.
    priority_temperature: float = 5.0
    online_pretrain_steps: int = 50_000
    online_pretrain_inner_lr: float = 3e-4
    density_learning_rate: float = 3e-4
    n_tau: int = 25_000
    kappa: float = 0.3
    kappa_cool_steps: int = 150_000
    kappa_end: float = 0.9
    utd_ratio: int = 5


OFFICIAL_OPT_ENVS = {
    # MuJoCo tasks reported in the OPT paper.
    "halfcheetah-random-v2", "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2",
    "hopper-random-v2", "hopper-medium-v2", "hopper-medium-replay-v2",
    "walker2d-random-v2", "walker2d-medium-v2", "walker2d-medium-replay-v2",
    # All six AntMaze tasks are present in the official config tree.
    "antmaze-umaze-v2", "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2", "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2", "antmaze-large-diverse-v2",
    # OPT reports/ships cloned Adroit configurations.
    "pen-cloned-v1", "door-cloned-v1", "hammer-cloned-v1", "relocate-cloned-v1",
}

ADAPTED_OPT_ENVS = {
    # Required only to match the RLPD benchmark used in this project.
    "halfcheetah-medium-expert-v2", "hopper-medium-expert-v2", "walker2d-medium-expert-v2",
    "pen-human-v1", "pen-expert-v1",
    "door-human-v1", "door-expert-v1",
    "hammer-human-v1", "hammer-expert-v1",
    "relocate-human-v1", "relocate-expert-v1",
}

SUPPORTED_OPT_ENVS = OFFICIAL_OPT_ENVS | ADAPTED_OPT_ENVS


def _apply_environment_config(config: TrainConfig) -> None:
    """Resolve OPT/backbone hyperparameters from env_name.

    Precedence rule for formal baselines:
      paper (especially Appendix-H kappa schedule) > released per-task YAML >
      fixed same-domain transfer for tasks absent from the OPT paper.

    Experiment protocol fields (offline_steps/online_steps/evaluation/seeds/device)
    are deliberately not changed here.
    """
    env = config.env_name.lower()
    if env not in SUPPORTED_OPT_ENVS:
        supported = "\n  ".join(sorted(SUPPORTED_OPT_ENVS))
        raise ValueError(
            f"Unsupported OPT environment: {config.env_name}.\n"
            f"This file intentionally supports the RLPD-matched 30-task suite only:\n  {supported}"
        )

    is_official = env in OFFICIAL_OPT_ENVS
    config.config_source = "official" if is_official else "adapted"
    if config.strict_official_only and not is_official:
        raise ValueError(
            f"{config.env_name} was not part of the OPT paper's reported/configured task set. "
            "Use strict_official_only=False to run the fixed RLPD-benchmark adaptation."
        )
    if not config.auto_env_config:
        config.config_source += ":manual_override"
        return

    # Parameters common to the released OPT implementation.
    config.hidden_dim = 256
    config.critic_learning_rate = 3e-4
    config.batch_size = 256
    config.replay_buffer_size = 2_000_000
    config.priority_replay_buffer_size = 1_000_000
    config.discount = 0.99
    config.tau = 5e-3
    config.exploration_noise = 0.1
    config.policy_noise = 0.2
    config.noise_clip = 0.5
    config.policy_frequency = 2
    config.normalize_states = False
    config.priority_temperature = 5.0
    config.online_pretrain_steps = 50_000
    config.online_pretrain_inner_lr = 3e-4
    config.density_learning_rate = 3e-4
    config.n_tau = 25_000

    # MuJoCo: TD3+BC backbone and UTD=5 (Appendix H).
    if any(token in env for token in LOCOMOTION_ENVS):
        config.algo_type = "TD3"
        config.actor_learning_rate = 3e-4
        config.actor_layer_norm = False
        config.critic_layer_norm = False
        config.actor_init_w = None
        config.critic_init_w = None
        config.normalize_rewards = False
        config.td3_bc_alpha = 2.5
        config.utd_ratio = 5

        # Appendix H, Table 15. κ is the coefficient on Q_on-pt.
        if "-random-" in env:
            config.kappa = 1.0
            config.kappa_cool_steps = 0
            config.kappa_end = 1.0
        elif "medium-replay" in env:
            config.kappa = 0.1
            config.kappa_cool_steps = 150_000
            config.kappa_end = 0.9
        elif "medium-expert" in env:
            # Not evaluated by OPT. Fixed transfer rule: use the reported
            # medium schedule for every locomotion medium-expert task; do not
            # tune per environment or per result.
            config.kappa = 0.3
            config.kappa_cool_steps = 150_000
            config.kappa_end = 0.9
        else:  # medium-v2
            config.kappa = 0.3
            config.kappa_cool_steps = 150_000
            config.kappa_end = 0.9
        return

    # AntMaze: released SPOT+OPT backbone.
    if "antmaze" in env:
        config.algo_type = "SPOT"
        config.actor_learning_rate = 1e-4
        config.actor_layer_norm = False
        config.critic_layer_norm = False
        config.actor_init_w = 1e-3
        config.critic_init_w = 3e-3
        config.normalize_rewards = True
        config.vae_learning_rate = 1e-3
        config.vae_hidden_dim = 750
        config.vae_beta = 0.5
        config.vae_steps = 100_000
        config.support_num_samples = 1
        config.support_use_iwae = False
        config.support_lambda_cool = True
        config.support_lambda_end = 0.2
        config.online_discount = 0.995
        config.utd_ratio = 1
        config.kappa = 0.1
        config.kappa_end = 0.9
        config.kappa_cool_steps = 200_000 if "large-diverse" in env else 100_000

        # Released SPOT+OPT YAMLs use a weaker support penalty as maze size grows.
        if "umaze" in env:
            config.support_lambda = 0.25
        elif "medium" in env:
            config.support_lambda = 0.05
        else:  # large
            config.support_lambda = 0.025
        return

    # Adroit: released OPT uses SPOT with layer normalization. The paper only
    # reports cloned datasets. For human/expert we transfer the identical
    # domain configuration, without task- or result-specific tuning.
    if any(token in env for token in ("pen", "door", "hammer", "relocate")):
        config.algo_type = "SPOT"
        config.actor_learning_rate = 1e-4
        config.actor_layer_norm = True
        config.critic_layer_norm = True
        config.actor_init_w = 1e-3
        config.critic_init_w = 3e-3
        config.normalize_rewards = False
        config.vae_learning_rate = 1e-3
        config.vae_hidden_dim = 750
        config.vae_beta = 0.5
        config.vae_steps = 100_000
        config.support_lambda = 1.0
        config.support_num_samples = 1
        config.support_use_iwae = False
        config.support_lambda_cool = True
        config.support_lambda_end = 0.5
        config.online_discount = 0.99
        config.utd_ratio = 1
        config.kappa = 0.1
        config.kappa_cool_steps = 250_000
        config.kappa_end = 0.9
        return

    raise AssertionError(f"Environment classification failed: {config.env_name}")


# -----------------------------------------------------------------------------
# Environment and dataset compatibility
# -----------------------------------------------------------------------------


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
        next_state, reward, terminated, truncated, info = output
        return (
            next_state,
            float(reward),
            bool(terminated or truncated),
            bool(terminated),
            bool(truncated),
            dict(info),
        )
    next_state, reward, done, info = output
    info = dict(info)
    truncated = bool(info.get("TimeLimit.truncated", False))
    terminated = bool(done and not truncated)
    return next_state, float(reward), bool(done), terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def observation(self, observation):
        return ((np.asarray(observation) - self.mean) / self.std).astype(np.float32)


def _unwrap_attr(env: gym.Env, attribute: str):
    candidate = env
    for _ in range(32):
        if hasattr(candidate, attribute):
            return getattr(candidate, attribute)
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    return None


def _max_episode_steps(env: gym.Env) -> int:
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
        return int(spec.max_episode_steps)
    value = _unwrap_attr(env, "_max_episode_steps")
    return int(value) if value is not None else 1_000


def _get_normalized_score(env: gym.Env, raw_return: float) -> Optional[float]:
    candidate = env
    for _ in range(32):
        if hasattr(candidate, "get_normalized_score"):
            try:
                return float(candidate.get_normalized_score(raw_return) * 100.0)
            except Exception:
                return None
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    return None


def _is_goal_reached(reward: float, info: Mapping) -> bool:
    if "goal_achieved" in info:
        return bool(info["goal_achieved"])
    return reward > 0.0


def _return_reward_range(
    dataset: Mapping[str, np.ndarray], max_episode_steps: int
) -> Tuple[float, float]:
    returns = []
    episode_return = 0.0
    episode_length = 0
    terminals = np.asarray(dataset["terminals"]).reshape(-1)
    rewards = np.asarray(dataset["rewards"]).reshape(-1)
    for reward, terminal in zip(rewards, terminals):
        episode_return += float(reward)
        episode_length += 1
        if bool(terminal) or episode_length == max_episode_steps:
            returns.append(episode_return)
            episode_return = 0.0
            episode_length = 0
    if episode_length > 0:
        returns.append(episode_return)
    if not returns:
        raise ValueError("Could not reconstruct trajectories for reward scaling.")
    return float(min(returns)), float(max(returns))


def _prepare_dataset(
    env: gym.Env,
    env_name: str,
    normalize_states: bool,
    normalize_rewards: bool,
) -> Tuple[NumpyBatch, np.ndarray, np.ndarray, Dict[str, float]]:
    dataset = {
        key: np.asarray(value).copy()
        for key, value in d4rl.qlearning_dataset(env).items()
    }

    observations = np.asarray(dataset["observations"], dtype=np.float32)
    next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    rewards = np.asarray(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
    terminals = np.asarray(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

    if normalize_states:
        state_mean = observations.mean(axis=0).astype(np.float32)
        state_std = (observations.std(axis=0) + 1e-3).astype(np.float32)
    else:
        state_mean = np.zeros(observations.shape[1:], dtype=np.float32)
        state_std = np.ones(observations.shape[1:], dtype=np.float32)

    observations = ((observations - state_mean) / state_std).astype(np.float32)
    next_observations = ((next_observations - state_mean) / state_std).astype(
        np.float32
    )

    reward_info: Dict[str, float] = {}
    lower_name = env_name.lower()
    if normalize_rewards:
        if any(domain in lower_name for domain in LOCOMOTION_ENVS):
            temporary = {
                "rewards": rewards.reshape(-1),
                "terminals": terminals.reshape(-1),
            }
            minimum_return, maximum_return = _return_reward_range(
                temporary, _max_episode_steps(env)
            )
            denominator = maximum_return - minimum_return
            if abs(denominator) < 1e-12:
                raise ValueError("Reward-return range is zero.")
            max_steps = _max_episode_steps(env)
            rewards = rewards / denominator * max_steps
            reward_info = {
                "minimum_return": minimum_return,
                "maximum_return": maximum_return,
                "max_episode_steps": float(max_steps),
            }
        elif "antmaze" in lower_name:
            rewards = rewards - 1.0

    batch: NumpyBatch = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards.astype(np.float32),
        "not_dones": (1.0 - terminals).astype(np.float32),
    }
    return batch, state_mean, state_std, reward_info


def _transform_online_reward(
    reward: float,
    env_name: str,
    normalize_rewards: bool,
    reward_info: Mapping[str, float],
) -> float:
    if not normalize_rewards:
        return float(reward)
    lower_name = env_name.lower()
    if any(domain in lower_name for domain in LOCOMOTION_ENVS):
        denominator = reward_info["maximum_return"] - reward_info["minimum_return"]
        return float(reward / denominator * reward_info["max_episode_steps"])
    if "antmaze" in lower_name:
        return float(reward - 1.0)
    return float(reward)


# -----------------------------------------------------------------------------
# Replay buffers and density-ratio prioritized replay
# -----------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
        seed: int,
    ):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.device = device
        self.size = 0
        self.pointer = 0
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def load_dataset(self, dataset: Mapping[str, np.ndarray]) -> None:
        size = len(dataset["observations"])
        if size > self.capacity:
            raise ValueError(
                f"Dataset size {size} exceeds replay capacity {self.capacity}."
            )
        self.observations[:size] = dataset["observations"]
        self.actions[:size] = dataset["actions"]
        self.next_observations[:size] = dataset["next_observations"]
        self.rewards[:size] = np.asarray(dataset["rewards"]).reshape(-1, 1)
        self.not_dones[:size] = np.asarray(dataset["not_dones"]).reshape(-1, 1)
        self.size = size
        self.pointer = size % self.capacity

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        reward: float,
        terminated: bool,
    ) -> None:
        index = self.pointer
        self.observations[index] = observation
        self.actions[index] = action
        self.next_observations[index] = next_observation
        self.rewards[index] = reward
        self.not_dones[index] = 1.0 - float(terminated)
        self.pointer = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TorchBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return self._indices_to_batch(indices)

    def _indices_to_batch(self, indices: np.ndarray) -> TorchBatch:
        return {
            "observations": torch.as_tensor(
                self.observations[indices], device=self.device
            ),
            "actions": torch.as_tensor(self.actions[indices], device=self.device),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], device=self.device
            ),
            "rewards": torch.as_tensor(self.rewards[indices], device=self.device),
            "not_dones": torch.as_tensor(
                self.not_dones[indices], device=self.device
            ),
        }


class SumTree:
    """Array-based sum tree used only for replay sampling priorities."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.leaf_offset = tree_capacity
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, data_index: int, priority: float) -> None:
        tree_index = self.leaf_offset + int(data_index)
        change = float(priority) - self.tree[tree_index]
        while tree_index >= 1:
            self.tree[tree_index] += change
            tree_index //= 2

    def find_prefixsum_index(self, mass: float) -> int:
        if self.total <= 0.0:
            raise ValueError("Cannot sample from a zero-priority tree.")
        mass = min(max(float(mass), 0.0), np.nextafter(self.total, 0.0))
        index = 1
        while index < self.leaf_offset:
            left = 2 * index
            if mass < self.tree[left]:
                index = left
            else:
                mass -= self.tree[left]
                index = left + 1
        return index - self.leaf_offset


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
        seed: int,
    ):
        super().__init__(observation_dim, action_dim, capacity, device, seed)
        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0

    def load_dataset(self, dataset: Mapping[str, np.ndarray]) -> None:
        super().load_dataset(dataset)
        for index in range(self.size):
            self.sum_tree.update(index, 1.0)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        reward: float,
        terminated: bool,
    ) -> None:
        index = self.pointer
        super().add(observation, action, next_observation, reward, terminated)
        self.sum_tree.update(index, self.max_priority)

    def sample(self, batch_size: int) -> Tuple[TorchBatch, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        total = self.sum_tree.total
        if total <= 0.0:
            raise RuntimeError("Priority sum is non-positive.")
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            mass = self.rng.uniform(segment * i, segment * (i + 1))
            index = self.sum_tree.find_prefixsum_index(mass)
            # The tree may contain stale leaves only if capacity handling is wrong.
            if index >= self.size and self.size < self.capacity:
                index = int(self.rng.integers(0, self.size))
            indices[i] = index
        return self._indices_to_batch(indices), indices

    def update_priorities(
        self, indices: np.ndarray, priorities: np.ndarray
    ) -> None:
        flat_indices = np.asarray(indices).reshape(-1)
        flat_priorities = np.asarray(priorities).reshape(-1)
        if len(flat_indices) != len(flat_priorities):
            raise ValueError("indices and priorities must have the same length.")
        for index, priority in zip(flat_indices, flat_priorities):
            value = float(priority)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid replay priority: {value}")
            self.sum_tree.update(int(index), value)
            self.max_priority = max(self.max_priority, value)


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------


def _initialize_head(layer: nn.Linear, init_w: Optional[float]) -> None:
    if init_w is not None:
        nn.init.uniform_(layer.weight, -init_w, init_w)
        nn.init.uniform_(layer.bias, -init_w, init_w)


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float,
        layer_norm: bool,
        init_w: Optional[float],
    ):
        super().__init__()
        head = nn.Linear(hidden_dim, action_dim)
        _initialize_head(head, init_w)
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            head,
            nn.Tanh(),
        )
        self.max_action = float(max_action)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(observations)

    @torch.no_grad()
    def act(self, observation: np.ndarray, device: torch.device) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).reshape(1, -1)
        return self(tensor).squeeze(0).cpu().numpy()


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        layer_norm: bool,
        init_w: Optional[float],
    ):
        super().__init__()
        head = nn.Linear(hidden_dim, 1)
        _initialize_head(head, init_w)
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            head,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1))


class DensityRatio(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat([observations, actions], dim=-1)
        values = F.relu(self.linear1(values))
        values = F.relu(self.linear2(values))
        return F.relu(self.linear3(values))


class VAE(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: Optional[int],
        max_action: float,
        hidden_dim: int,
    ):
        super().__init__()
        self.latent_dim = 2 * action_dim if latent_dim is None else int(latent_dim)
        self.max_action = float(max_action)
        self.encoder_shared = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, self.latent_dim)
        self.log_std = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(observation_dim + self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_shared(torch.cat([observations, actions], dim=-1))
        mean = self.mean(hidden)
        std = self.log_std(hidden).clamp(-4.0, 15.0).exp()
        return mean, std

    def decode(
        self, observations: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if latent is None:
            latent = torch.randn(
                (observations.shape[0], self.latent_dim), device=self.device
            ).clamp(-0.5, 0.5)
        return self.max_action * self.decoder(
            torch.cat([observations, latent], dim=-1)
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(observations, actions)
        latent = mean + std * torch.randn_like(std)
        reconstruction = self.decode(observations, latent)
        return reconstruction, mean, std

    def importance_sampling_estimator(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        beta: float,
        num_samples: int,
    ) -> torch.Tensor:
        mean, std = self.encode(observations, actions)
        mean_samples = mean[:, None, :].expand(-1, num_samples, -1)
        std_samples = std[:, None, :].expand(-1, num_samples, -1)
        latent = mean_samples + std_samples * torch.randn_like(std_samples)
        observation_samples = observations[:, None, :].expand(
            -1, num_samples, -1
        )
        action_samples = actions[:, None, :].expand(-1, num_samples, -1)
        decoded_mean = self.decode(observation_samples, latent)
        decoder_std = torch.full_like(decoded_mean, math.sqrt(beta / 4.0))
        log_q = td.Normal(mean_samples, std_samples).log_prob(latent).sum(dim=-1)
        log_p = td.Normal(
            torch.zeros_like(latent), torch.ones_like(latent)
        ).log_prob(latent).sum(dim=-1)
        log_px = td.Normal(decoded_mean, decoder_std).log_prob(
            action_samples
        ).sum(dim=-1)
        weights = log_px + log_p - log_q
        return torch.logsumexp(weights, dim=-1) - math.log(num_samples)


class HotPlug:
    """Differentiable one-step parameter replacement from official OPT code."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.parameters = OrderedDict(model.named_parameters())

    def update(self, learning_rate: float) -> None:
        for name, parameter in self.parameters.items():
            path = name.split(".")
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if learning_rate > 0.0 and parameter.requires_grad:
                if parameter.grad is None:
                    raise RuntimeError(f"Missing inner-loop gradient for {name}.")
                cursor._parameters[path[-1]] = (
                    parameter - learning_rate * parameter.grad
                )
            else:
                cursor._parameters[path[-1]] = parameter

    def restore(self) -> None:
        self.update(0.0)


# -----------------------------------------------------------------------------
# OPT learner
# -----------------------------------------------------------------------------


class OPTLearner:
    def __init__(
        self,
        config: TrainConfig,
        actor: Actor,
        offline_critic_1: Critic,
        offline_critic_2: Critic,
        online_critic_1: Critic,
        online_critic_2: Critic,
        density_ratio: DensityRatio,
        vae: Optional[VAE],
        max_action: float,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.max_action = float(max_action)
        self.actor = actor
        self.actor_target = deepcopy(actor).to(device)
        self.offline_critic_1 = offline_critic_1
        self.offline_critic_2 = offline_critic_2
        self.offline_critic_1_target = deepcopy(offline_critic_1).to(device)
        self.offline_critic_2_target = deepcopy(offline_critic_2).to(device)
        self.online_critic_1 = online_critic_1
        self.online_critic_2 = online_critic_2
        self.online_critic_1_target = deepcopy(online_critic_1).to(device)
        self.online_critic_2_target = deepcopy(online_critic_2).to(device)
        self.density_ratio = density_ratio
        self.vae = vae

        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=config.actor_learning_rate
        )
        self.offline_critic_1_optimizer = torch.optim.Adam(
            offline_critic_1.parameters(), lr=config.critic_learning_rate
        )
        self.offline_critic_2_optimizer = torch.optim.Adam(
            offline_critic_2.parameters(), lr=config.critic_learning_rate
        )
        self.online_critic_1_optimizer = torch.optim.Adam(
            online_critic_1.parameters(), lr=config.critic_learning_rate
        )
        self.online_critic_2_optimizer = torch.optim.Adam(
            online_critic_2.parameters(), lr=config.critic_learning_rate
        )
        self.density_optimizer = torch.optim.Adam(
            density_ratio.parameters(), lr=config.density_learning_rate
        )
        self.vae_optimizer = (
            torch.optim.Adam(vae.parameters(), lr=config.vae_learning_rate)
            if vae is not None
            else None
        )

        self.hotplug_1 = HotPlug(self.online_critic_1)
        self.hotplug_2 = HotPlug(self.online_critic_2)
        self.total_updates = 0
        self.online_updates = 0
        self.online_pretrain_updates = 0

    def reset_online_optimizers(self) -> None:
        """Match the released transition from offline to online training."""
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.offline_critic_1_optimizer = torch.optim.Adam(
            self.offline_critic_1.parameters(),
            lr=self.config.critic_learning_rate,
        )
        self.offline_critic_2_optimizer = torch.optim.Adam(
            self.offline_critic_2.parameters(),
            lr=self.config.critic_learning_rate,
        )

    def set_online_discount(self) -> None:
        if self.config.algo_type.upper() == "SPOT":
            self.config.discount = self.config.online_discount

    def _target_action(
        self, next_observations: torch.Tensor, reference_actions: torch.Tensor
    ) -> torch.Tensor:
        noise = (
            torch.randn_like(reference_actions)
            * self.config.policy_noise
            * self.max_action
        ).clamp(
            -self.config.noise_clip * self.max_action,
            self.config.noise_clip * self.max_action,
        )
        return (self.actor_target(next_observations) + noise).clamp(
            -self.max_action, self.max_action
        )

    def _td_target(
        self,
        batch: TorchBatch,
        target_critic_1: Critic,
        target_critic_2: Critic,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action = self._target_action(
                batch["next_observations"], batch["actions"]
            )
            target_q = torch.minimum(
                target_critic_1(batch["next_observations"], next_action),
                target_critic_2(batch["next_observations"], next_action),
            )
            return batch["rewards"] + (
                batch["not_dones"] * self.config.discount * target_q
            )

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    source_parameter, alpha=tau
                )

    def _update_critic_pair(
        self,
        batch: TorchBatch,
        critic_1: Critic,
        critic_2: Critic,
        target_critic_1: Critic,
        target_critic_2: Critic,
        optimizer_1: torch.optim.Optimizer,
        optimizer_2: torch.optim.Optimizer,
    ) -> torch.Tensor:
        target_q = self._td_target(batch, target_critic_1, target_critic_2)
        loss = F.mse_loss(
            critic_1(batch["observations"], batch["actions"]), target_q
        ) + F.mse_loss(
            critic_2(batch["observations"], batch["actions"]), target_q
        )
        optimizer_1.zero_grad(set_to_none=True)
        optimizer_2.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        return loss.detach()

    def vae_update(self, batch: TorchBatch) -> Dict[str, float]:
        if self.vae is None or self.vae_optimizer is None:
            raise RuntimeError("VAE updates require algo_type=SPOT.")
        reconstruction, mean, std = self.vae(
            batch["observations"], batch["actions"]
        )
        reconstruction_loss = F.mse_loss(reconstruction, batch["actions"])
        kl_loss = -0.5 * (
            1.0 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)
        ).mean()
        vae_loss = reconstruction_loss + self.config.vae_beta * kl_loss
        self.vae_optimizer.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_optimizer.step()
        return {
            "vae/reconstruction_loss": float(reconstruction_loss.detach().item()),
            "vae/kl_loss": float(kl_loss.detach().item()),
            "vae/loss": float(vae_loss.detach().item()),
        }

    def _spot_elbo_loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("SPOT loss requires a trained VAE.")
        num_samples = self.config.support_num_samples
        mean, std = self.vae.encode(observations, actions)
        mean_samples = mean[:, None, :].expand(-1, num_samples, -1)
        std_samples = std[:, None, :].expand(-1, num_samples, -1)
        latent = mean_samples + std_samples * torch.randn_like(std_samples)
        observation_samples = observations[:, None, :].expand(
            -1, num_samples, -1
        )
        action_samples = actions[:, None, :].expand(-1, num_samples, -1)
        reconstruction = self.vae.decode(observation_samples, latent)
        reconstruction_loss = ((reconstruction - action_samples) ** 2).mean(
            dim=(1, 2)
        )
        kl_loss = -0.5 * (
            1.0
            + torch.log(std.pow(2))
            - mean.pow(2)
            - std.pow(2)
        ).mean(dim=-1)
        return reconstruction_loss + self.config.vae_beta * kl_loss

    def _spot_negative_log_density(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("SPOT loss requires a trained VAE.")
        if self.config.support_use_iwae:
            return -self.vae.importance_sampling_estimator(
                observations,
                actions,
                self.config.vae_beta,
                self.config.support_num_samples,
            )
        return self._spot_elbo_loss(observations, actions)

    def _support_lambda(self) -> float:
        if not self.config.support_lambda_cool:
            return self.config.support_lambda
        fraction = max(
            self.config.support_lambda_end,
            1.0 - self.online_updates / max(1, self.config.online_steps),
        )
        return self.config.support_lambda * fraction

    def offline_update(self, batch: TorchBatch) -> Dict[str, float]:
        self.total_updates += 1
        critic_loss = self._update_critic_pair(
            batch,
            self.offline_critic_1,
            self.offline_critic_2,
            self.offline_critic_1_target,
            self.offline_critic_2_target,
            self.offline_critic_1_optimizer,
            self.offline_critic_2_optimizer,
        )
        metrics = {"offline/critic_loss": float(critic_loss.item())}

        if self.total_updates % self.config.policy_frequency == 0:
            policy_action = self.actor(batch["observations"])
            q_value = self.offline_critic_1(
                batch["observations"], policy_action
            )
            if self.config.algo_type.upper() == "TD3":
                scale = self.config.td3_bc_alpha / (
                    q_value.abs().mean().detach() + 1e-10
                )
                actor_loss = -scale * q_value.mean() + F.mse_loss(
                    policy_action, batch["actions"]
                )
                metrics["offline/td3_bc_scale"] = float(scale.item())
            elif self.config.algo_type.upper() == "SPOT":
                negative_log_density = self._spot_negative_log_density(
                    batch["observations"], policy_action
                )
                scale = 1.0 / (q_value.abs().mean().detach() + 1e-10)
                support_lambda = self._support_lambda()
                actor_loss = -scale * q_value.mean() + (
                    support_lambda * negative_log_density.mean()
                )
                metrics.update(
                    {
                        "offline/support_lambda": float(support_lambda),
                        "offline/negative_log_density": float(
                            negative_log_density.detach().mean().item()
                        ),
                    }
                )
            else:
                raise ValueError("algo_type must be TD3 or SPOT.")

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            metrics["offline/actor_loss"] = float(actor_loss.detach().item())

            self._soft_update(
                self.offline_critic_1_target,
                self.offline_critic_1,
                self.config.tau,
            )
            self._soft_update(
                self.offline_critic_2_target,
                self.offline_critic_2,
                self.config.tau,
            )
            self._soft_update(
                self.actor_target, self.actor, self.config.tau
            )
        return metrics

    def online_pretrain(
        self,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        num_steps: int,
    ) -> Dict[str, float]:
        if online_buffer.size == 0:
            raise ValueError("Online pre-training requires collected online data.")
        final_metrics: Dict[str, float] = {}
        iterator = trange(num_steps, desc="OPT online pre-training")
        for _ in iterator:
            self.online_pretrain_updates += 1
            offline_batch = offline_buffer.sample(self.config.batch_size)
            offline_target = self._td_target(
                offline_batch,
                self.online_critic_1_target,
                self.online_critic_2_target,
            )
            offline_loss = F.mse_loss(
                self.online_critic_1(
                    offline_batch["observations"], offline_batch["actions"]
                ),
                offline_target,
            ) + F.mse_loss(
                self.online_critic_2(
                    offline_batch["observations"], offline_batch["actions"]
                ),
                offline_target,
            )

            self.online_critic_1_optimizer.zero_grad(set_to_none=True)
            self.online_critic_2_optimizer.zero_grad(set_to_none=True)
            # Official implementation intentionally uses backward(create_graph=True)
            # before replacing parameters with a differentiable inner step.
            offline_loss.backward(create_graph=True)
            self.hotplug_1.update(self.config.online_pretrain_inner_lr)
            self.hotplug_2.update(self.config.online_pretrain_inner_lr)

            online_batch = online_buffer.sample(self.config.batch_size)
            online_target = self._td_target(
                online_batch,
                self.online_critic_1_target,
                self.online_critic_2_target,
            )
            online_loss = F.mse_loss(
                self.online_critic_1(
                    online_batch["observations"], online_batch["actions"]
                ),
                online_target,
            ) + F.mse_loss(
                self.online_critic_2(
                    online_batch["observations"], online_batch["actions"]
                ),
                online_target,
            )

            gradient_weight = torch.minimum(
                offline_loss.detach(), online_loss.detach()
            ) / (online_loss.detach() + 1e-10)
            normalized_online_loss = gradient_weight * online_loss
            normalized_online_loss.backward()
            self.online_critic_1_optimizer.step()
            self.online_critic_2_optimizer.step()
            self.hotplug_1.restore()
            self.hotplug_2.restore()
            # Break the create_graph=True parameter/gradient reference cycle.
            # This does not change the optimizer step and prevents accumulation
            # across the 50k online-pretraining iterations.
            for parameter in self.online_critic_1.parameters():
                parameter.grad = None
            for parameter in self.online_critic_2.parameters():
                parameter.grad = None

            if (
                self.online_pretrain_updates
                % self.config.policy_frequency
                == 0
            ):
                self._soft_update(
                    self.online_critic_1_target,
                    self.online_critic_1,
                    self.config.tau,
                )
                self._soft_update(
                    self.online_critic_2_target,
                    self.online_critic_2,
                    self.config.tau,
                )

            final_metrics = {
                "online_pretrain/offline_td_loss": float(
                    offline_loss.detach().item()
                ),
                "online_pretrain/online_td_loss": float(
                    online_loss.detach().item()
                ),
                "online_pretrain/gradient_weight": float(
                    gradient_weight.detach().item()
                ),
            }
        return final_metrics

    def _density_ratio_update(
        self,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        offline_batch = offline_buffer.sample(self.config.batch_size)
        online_batch = online_buffer.sample(self.config.batch_size)
        offline_weight = self.density_ratio(
            offline_batch["observations"], offline_batch["actions"]
        )
        online_weight = self.density_ratio(
            online_batch["observations"], online_batch["actions"]
        )
        offline_f_star = -torch.log(2.0 / (offline_weight + 1.0) + 1e-10)
        online_f_prime = torch.log(
            2.0 * online_weight / (online_weight + 1.0) + 1e-10
        )
        density_loss = (offline_f_star - online_f_prime).mean()
        self.density_optimizer.zero_grad(set_to_none=True)
        density_loss.backward()
        self.density_optimizer.step()
        return offline_weight.detach(), {
            "online/density_loss": float(density_loss.detach().item()),
            "online/offline_density": float(offline_weight.detach().mean().item()),
            "online/online_density": float(online_weight.detach().mean().item()),
        }

    def online_update(
        self,
        priority_buffer: PriorityReplayBuffer,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        kappa: float,
    ) -> Dict[str, float]:
        self.online_updates += 1
        self.total_updates += 1

        offline_density, metrics = self._density_ratio_update(
            offline_buffer, online_buffer
        )
        batch, indices = priority_buffer.sample(self.config.batch_size)
        with torch.no_grad():
            sampled_density = self.density_ratio(
                batch["observations"], batch["actions"]
            )
            inverse_temperature = 1.0 / self.config.priority_temperature
            normalization = offline_density.pow(inverse_temperature).mean()
            priorities = sampled_density.pow(inverse_temperature) / (
                normalization + 1e-10
            )
            priorities = priorities.clamp(1e-3, 1e3)
        priority_buffer.update_priorities(
            indices, priorities.squeeze(-1).cpu().numpy()
        )

        offline_critic_loss = self._update_critic_pair(
            batch,
            self.offline_critic_1,
            self.offline_critic_2,
            self.offline_critic_1_target,
            self.offline_critic_2_target,
            self.offline_critic_1_optimizer,
            self.offline_critic_2_optimizer,
        )
        online_critic_loss = self._update_critic_pair(
            batch,
            self.online_critic_1,
            self.online_critic_2,
            self.online_critic_1_target,
            self.online_critic_2_target,
            self.online_critic_1_optimizer,
            self.online_critic_2_optimizer,
        )
        metrics.update(
            {
                "online/offline_critic_loss": float(offline_critic_loss.item()),
                "online/new_critic_loss": float(online_critic_loss.item()),
                "online/priority_mean": float(priorities.mean().item()),
                "online/kappa": float(kappa),
            }
        )

        if self.total_updates % self.config.policy_frequency == 0:
            policy_action = self.actor(batch["observations"])
            old_q = self.offline_critic_1(batch["observations"], policy_action)
            new_q = self.online_critic_1(batch["observations"], policy_action)
            blended_q = (1.0 - kappa) * old_q + kappa * new_q

            if self.config.algo_type.upper() == "TD3":
                actor_loss = -blended_q.mean()
            elif self.config.algo_type.upper() == "SPOT":
                negative_log_density = self._spot_negative_log_density(
                    batch["observations"], policy_action
                )
                support_lambda = self._support_lambda()
                scale = 1.0 / (blended_q.abs().mean().detach() + 1e-10)
                actor_loss = -scale * blended_q.mean() + (
                    support_lambda * negative_log_density.mean()
                )
                metrics.update(
                    {
                        "online/support_lambda": float(support_lambda),
                        "online/negative_log_density": float(
                            negative_log_density.detach().mean().item()
                        ),
                    }
                )
            else:
                raise ValueError("algo_type must be TD3 or SPOT.")

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            metrics.update(
                {
                    "online/actor_loss": float(actor_loss.detach().item()),
                    "online/old_q": float(old_q.detach().mean().item()),
                    "online/new_q": float(new_q.detach().mean().item()),
                    "online/blended_q": float(
                        blended_q.detach().mean().item()
                    ),
                }
            )

            self._soft_update(
                self.offline_critic_1_target,
                self.offline_critic_1,
                self.config.tau,
            )
            self._soft_update(
                self.offline_critic_2_target,
                self.offline_critic_2,
                self.config.tau,
            )
            self._soft_update(
                self.online_critic_1_target,
                self.online_critic_1,
                self.config.tau,
            )
            self._soft_update(
                self.online_critic_2_target,
                self.online_critic_2,
                self.config.tau,
            )
            self._soft_update(
                self.actor_target, self.actor, self.config.tau
            )
        return metrics

    def state_dict(self) -> Dict:
        state = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "offline_critic_1": self.offline_critic_1.state_dict(),
            "offline_critic_2": self.offline_critic_2.state_dict(),
            "offline_critic_1_target": self.offline_critic_1_target.state_dict(),
            "offline_critic_2_target": self.offline_critic_2_target.state_dict(),
            "offline_critic_1_optimizer": self.offline_critic_1_optimizer.state_dict(),
            "offline_critic_2_optimizer": self.offline_critic_2_optimizer.state_dict(),
            "online_critic_1": self.online_critic_1.state_dict(),
            "online_critic_2": self.online_critic_2.state_dict(),
            "online_critic_1_target": self.online_critic_1_target.state_dict(),
            "online_critic_2_target": self.online_critic_2_target.state_dict(),
            "online_critic_1_optimizer": self.online_critic_1_optimizer.state_dict(),
            "online_critic_2_optimizer": self.online_critic_2_optimizer.state_dict(),
            "density_ratio": self.density_ratio.state_dict(),
            "density_optimizer": self.density_optimizer.state_dict(),
            "total_updates": self.total_updates,
            "online_updates": self.online_updates,
            "online_pretrain_updates": self.online_pretrain_updates,
        }
        if self.vae is not None and self.vae_optimizer is not None:
            state["vae"] = self.vae.state_dict()
            state["vae_optimizer"] = self.vae_optimizer.state_dict()
        return state


# -----------------------------------------------------------------------------
# Evaluation and schedules
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    actor: Actor,
    env: gym.Env,
    device: torch.device,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    actor.eval()
    returns = []
    lengths = []
    successes = []
    for episode in range(episodes):
        observation = _reset_env(env, seed + episode)
        done = False
        episode_return = 0.0
        episode_length = 0
        goal_reached = False
        while not done:
            action = actor.act(observation, device)
            observation, reward, done, _, _, info = _step_env(env, action)
            episode_return += reward
            episode_length += 1
            if not goal_reached:
                goal_reached = _is_goal_reached(reward, info)
        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(float(goal_reached))
    actor.train()
    raw_return = float(np.mean(returns))
    result = {
        "return": raw_return,
        "length": float(np.mean(lengths)),
        "success": float(np.mean(successes)),
    }
    normalized_score = _get_normalized_score(env, raw_return)
    if normalized_score is not None:
        result["normalized_score"] = normalized_score
    return result


def _kappa_at_step(config: TrainConfig, online_step: int) -> float:
    """Linear schedule used by the released configs after N_tau collection."""
    if config.kappa_cool_steps <= 0:
        return float(config.kappa)
    if config.kappa_cool_steps <= config.n_tau:
        return float(config.kappa_end)
    fraction = (online_step - config.n_tau) / (
        config.kappa_cool_steps - config.n_tau
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(config.kappa + fraction * (config.kappa_end - config.kappa))


def _log_metrics(
    writer: SummaryWriter, metrics: Mapping[str, float], step: int
) -> None:
    for key, value in metrics.items():
        if math.isfinite(float(value)):
            writer.add_scalar(key, float(value), step)


def _validate_config(config: TrainConfig) -> None:
    if config.algo_type.upper() not in {"TD3", "SPOT"}:
        raise ValueError("algo_type must be TD3 or SPOT.")
    if config.batch_size < 1 or config.utd_ratio < 1:
        raise ValueError("batch_size and utd_ratio must be positive.")
    if config.n_tau < 1 or config.online_pretrain_steps < 1:
        raise ValueError("n_tau and online_pretrain_steps must be positive.")
    if config.n_tau > config.online_steps:
        raise ValueError("n_tau cannot exceed online_steps.")
    if config.priority_replay_buffer_size < 1:
        raise ValueError("priority_replay_buffer_size must be positive.")
    if config.priority_temperature <= 0.0:
        raise ValueError("priority_temperature must be positive.")
    if not 0.0 <= config.kappa <= 1.0:
        raise ValueError("kappa must be in [0,1].")
    if not 0.0 <= config.kappa_end <= 1.0:
        raise ValueError("kappa_end must be in [0,1].")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(config: TrainConfig) -> None:
    _apply_environment_config(config)
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)
    device = torch.device(config.device)

    base_train_env = gym.make(config.env_name)
    base_eval_env = gym.make(config.env_name)
    _seed_env(base_train_env, config.train_seed)
    _seed_env(base_eval_env, config.eval_seed)

    dataset, state_mean, state_std, reward_info = _prepare_dataset(
        base_train_env,
        config.env_name,
        config.normalize_states,
        config.normalize_rewards,
    )
    train_env: gym.Env = base_train_env
    eval_env: gym.Env = base_eval_env
    if config.normalize_states:
        train_env = NormalizeObservation(train_env, state_mean, state_std)
        eval_env = NormalizeObservation(eval_env, state_mean, state_std)

    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise TypeError("OPT requires a flat continuous observation space.")
    if not isinstance(train_env.action_space, gym.spaces.Box):
        raise TypeError("OPT requires a continuous Box action space.")
    observation_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))
    max_action = float(np.asarray(train_env.action_space.high).reshape(-1)[0])

    offline_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        config.replay_buffer_size,
        device,
        config.train_seed + 1,
    )
    offline_buffer.load_dataset(dataset)
    online_buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        config.replay_buffer_size,
        device,
        config.train_seed + 2,
    )
    priority_buffer = PriorityReplayBuffer(
        observation_dim,
        action_dim,
        config.priority_replay_buffer_size,
        device,
        config.train_seed + 3,
    )
    priority_buffer.load_dataset(dataset)

    actor = Actor(
        observation_dim,
        action_dim,
        config.hidden_dim,
        max_action,
        config.actor_layer_norm,
        config.actor_init_w,
    ).to(device)
    critic_kwargs = dict(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        layer_norm=config.critic_layer_norm,
        init_w=config.critic_init_w,
    )
    offline_critic_1 = Critic(**critic_kwargs).to(device)
    offline_critic_2 = Critic(**critic_kwargs).to(device)
    online_critic_1 = Critic(**critic_kwargs).to(device)
    online_critic_2 = Critic(**critic_kwargs).to(device)
    density_ratio = DensityRatio(
        observation_dim, action_dim, config.hidden_dim
    ).to(device)
    vae = (
        VAE(
            observation_dim,
            action_dim,
            config.vae_latent_dim,
            max_action,
            config.vae_hidden_dim,
        ).to(device)
        if config.algo_type.upper() == "SPOT"
        else None
    )
    learner = OPTLearner(
        config,
        actor,
        offline_critic_1,
        offline_critic_2,
        online_critic_1,
        online_critic_2,
        density_ratio,
        vae,
        max_action,
        device,
    )

    run_name = (
        f"{config.env_name}_seed{config.train_seed}_"
        f"{config.algo_type.upper()}_OPT"
    )
    log_dir = os.path.join(
        config.log_root, config.env_name.split("-")[0], run_name
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"env={config.env_name}\n"
        f"device={device}\n"
        f"config_source={config.config_source}\n"
        f"base={config.algo_type.upper()}\n"
        f"offline_steps={config.offline_steps}\n"
        f"online_steps={config.online_steps}\n"
        f"N_tau={config.n_tau}, N_pretrain={config.online_pretrain_steps}\n"
        f"kappa={config.kappa}->{config.kappa_end}, "
        f"UTD={config.utd_ratio}\n"
        f"log_dir={log_dir}",
        flush=True,
    )
    if config.config_source.startswith("adapted"):
        print(
            "[NOTE] This environment was not reported by the OPT paper. "
            "The algorithm is OPT, but the environment hyperparameters use the "
            "fixed same-domain transfer rule documented in this file.",
            flush=True,
        )

    if config.algo_type.upper() == "SPOT":
        for step in trange(config.vae_steps, desc="SPOT VAE pre-training"):
            metrics = learner.vae_update(
                offline_buffer.sample(config.batch_size)
            )
            if step % config.log_every == 0:
                _log_metrics(writer, metrics, step)
        assert learner.vae is not None
        learner.vae.eval()

    for step in trange(config.offline_steps, desc="OPT offline pre-training"):
        metrics = learner.offline_update(
            offline_buffer.sample(config.batch_size)
        )
        if step % config.log_every == 0:
            _log_metrics(writer, metrics, step)
        if step % config.eval_every == 0:
            evaluation = evaluate(
                actor,
                eval_env,
                device,
                config.eval_episodes,
                config.eval_seed,
            )
            _log_metrics(
                writer,
                {f"offline_evaluation/{k}": v for k, v in evaluation.items()},
                step,
            )
            print(f"[offline eval] step={step}: {evaluation}", flush=True)
            writer.flush()

    learner.set_online_discount()
    learner.reset_online_optimizers()
    observation = _reset_env(train_env)
    episode_return = 0.0
    episode_length = 0
    goal_reached = False
    online_pretraining_done = False

    for online_step in trange(config.online_steps, desc="OPT online phase"):
        action = actor.act(observation, device)
        exploration_noise = np.random.normal(
            0.0,
            config.exploration_noise,
            size=action.shape,
        ).astype(np.float32)
        exploration_noise = np.clip(
            exploration_noise, -config.noise_clip, config.noise_clip
        )
        action = np.clip(
            max_action * (action + exploration_noise),
            -max_action,
            max_action,
        )

        (
            next_observation,
            raw_reward,
            done,
            terminated,
            _,
            info,
        ) = _step_env(train_env, action)
        training_reward = _transform_online_reward(
            raw_reward,
            config.env_name,
            config.normalize_rewards,
            reward_info,
        )
        online_buffer.add(
            observation,
            action,
            next_observation,
            training_reward,
            terminated,
        )
        priority_buffer.add(
            observation,
            action,
            next_observation,
            training_reward,
            terminated,
        )

        episode_return += raw_reward
        episode_length += 1
        if not goal_reached:
            goal_reached = _is_goal_reached(raw_reward, info)
        observation = next_observation

        if done:
            writer.add_scalar("training/return", episode_return, online_step)
            writer.add_scalar("training/length", episode_length, online_step)
            writer.add_scalar(
                "training/success", float(goal_reached), online_step
            )
            observation = _reset_env(train_env)
            episode_return = 0.0
            episode_length = 0
            goal_reached = False

        metrics: Dict[str, float] = {}
        if online_buffer.size >= config.n_tau:
            if not online_pretraining_done:
                pretrain_metrics = learner.online_pretrain(
                    offline_buffer,
                    online_buffer,
                    config.online_pretrain_steps,
                )
                _log_metrics(writer, pretrain_metrics, online_step)
                online_pretraining_done = True

            kappa = _kappa_at_step(config, online_step + 1)
            for _ in range(config.utd_ratio):
                metrics = learner.online_update(
                    priority_buffer,
                    offline_buffer,
                    online_buffer,
                    kappa,
                )

        if metrics and online_step % config.log_every == 0:
            _log_metrics(writer, metrics, online_step)
        if online_step % config.eval_every == 0:
            evaluation = evaluate(
                actor,
                eval_env,
                device,
                config.eval_episodes,
                config.eval_seed,
            )
            _log_metrics(
                writer,
                {f"evaluation/{k}": v for k, v in evaluation.items()},
                online_step,
            )
            print(f"[online eval] step={online_step}: {evaluation}", flush=True)
            writer.flush()

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "learner": learner.state_dict(),
                "config": asdict(config),
                "state_mean": state_mean,
                "state_std": state_std,
                "reward_info": reward_info,
            },
            os.path.join(checkpoint_dir, "final.pt"),
        )

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train(pyrallis.parse(config_class=TrainConfig))
