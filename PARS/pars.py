"""
Single-file PyTorch implementation of PARS for offline-to-online RL.

Algorithm source:
    Kim et al., "Penalizing Infeasible Actions and Reward Scaling in
    Reinforcement Learning with Offline Data", ICML 2025 (Spotlight).

Implementation source of truth:
    1) The PARS paper and appendix.
    2) The authors' official JAX implementation:
       https://github.com/LGAI-Research/pars

This file is a PyTorch reimplementation intended for the same D4RL benchmark
suite used by the user's RLPD experiments.

Important implementation choices:
- Reward Scaling + Layer Normalization (RS-LN).
- Penalizing infeasible actions (PA) by regressing their Q-values to Q_min.
- Ensemble critic and random critic subsampling for TD targets.
- TD3-style deterministic actor with behavior-cloning regularization offline.
- Offline/online mixed replay and UTD=20 during online fine-tuning.
- Dataset-specific PARS hyperparameters are automatically selected by env_name.
- Tasks not evaluated in the original PARS online-finetuning benchmark are
  clearly marked as "adapted" and use a fixed same-domain transfer rule.

Paper/code discrepancy handled deliberately:
The paper and released YAML configs define `online_beta`. The released JAX
`online_train` function passes `self.beta` to its online actor loss instead of
`self.online_beta`, leaving `online_beta` effectively unused. Because the paper
is the algorithmic authority and the configs explicitly tune online beta, this
implementation uses `online_beta` in the online actor objective.
"""

import math
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import d4rl  # noqa: F401
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainConfig:
    # Environment / reproducibility.
    env_name: str = "hopper-medium-replay-v2"
    train_seed: int = 0
    eval_seed: int = 100
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    deterministic_torch: bool = False

    # Automatically load the paper/official task configuration.
    auto_env_config: bool = True
    # If True, reject tasks that were not evaluated in PARS online fine-tuning.
    strict_official_only: bool = False

    # Experiment protocol. PARS uses 3M offline updates in the published setup.
    # The online interaction budget is deliberately standardized to 1M here.
    pretrain_steps: int = 3_000_000
    num_total_steps: int = 1_000_000
    eval_every: int = 5_000
    eval_episodes: int = 10
    log_every: int = 1_000
    batch_size: int = 256
    replay_buffer_size: int = 1_000_000

    # General PARS / TD3 hyperparameters.
    hidden_dim: int = 256
    critic_hidden_layers: int = 2
    actor_learning_rate: float = 3e-4
    online_actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 5e-3
    policy_noise: float = 0.0
    online_policy_noise: float = 0.0
    noise_clip: float = 0.5
    policy_frequency: int = 2
    exploration_noise: float = 0.1

    # RS-LN / PA.
    layer_norm: bool = True
    reward_scale: float = 1.0
    q_min: float = 0.0
    offline_alpha: float = 0.0
    offline_beta: float = 0.0
    online_alpha: float = 0.0
    online_beta: float = 0.0
    offline_ood_action_weight: float = 100.0
    # The released online JAX code uses a fixed multiplier of 100.
    online_ood_action_weight: float = 100.0

    # Ensemble settings.
    num_ensemble: int = 10
    num_sample_ensemble: int = 2
    online_num_sample_ensemble: int = 2
    online_num_action_sample_ensemble: int = 1

    # Online fine-tuning.
    utd_ratio: int = 20
    offline_ratio: float = 0.5
    learning_starts: int = 0

    # State normalization is supported, but official PARS configs use it off.
    normalize_states: bool = False

    # Actor cosine schedule: paper uses it on Adroit offline pre-training.
    actor_cosine_scheduler: bool = False
    scheduler_steps: Optional[int] = None
    min_actor_learning_rate: float = 0.0

    # Runtime output.
    log_root: str = "logs/PARS"
    checkpoints_path: Optional[str] = None

    # Filled automatically.
    config_source: str = "manual"


OFFICIAL_ONLINE_ENVS = {
    # MuJoCo online-finetuning tasks reported by PARS.
    "halfcheetah-random-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-random-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-random-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    # AntMaze.
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
    # Adroit online-finetuning tasks.
    "pen-cloned-v1",
    "door-cloned-v1",
    "hammer-cloned-v1",
    "relocate-cloned-v1",
}

SUPPORTED_ENVS = {
    # MuJoCo: 3 x 4.
    *{
        f"{env}-{dataset}-v2"
        for env in ("halfcheetah", "hopper", "walker2d")
        for dataset in ("random", "medium", "medium-replay", "medium-expert")
    },
    # Adroit: 4 x 3.
    *{
        f"{env}-{dataset}-v1"
        for env in ("pen", "door", "hammer", "relocate")
        for dataset in ("human", "cloned", "expert")
    },
    # AntMaze: 6.
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
}


# Paper Table 8.
Q_MIN_BY_DOMAIN = {
    "halfcheetah": -366.0,
    "hopper": -166.0,
    "walker2d": -229.0,
    "pen": -715.0,
    "door": -42.0,
    "hammer": -348.0,
    "relocate": 0.0,
    "antmaze": 0.0,
}


# Paper Table 11: MuJoCo.
# offline_alpha, policy_noise, S_kcritic, online_alpha, S_kactor
MUJOCO_TABLE = {
    "halfcheetah-random-v2":        (1e-4, 0.2, 2, 1e-4, 1),
    "halfcheetah-medium-v2":        (1e-4, 0.0, 2, 1e-4, 10),
    "halfcheetah-medium-replay-v2": (1e-4, 0.0, 2, 1e-4, 10),
    "halfcheetah-medium-expert-v2": (1e-4, 0.2, 10, None, None),
    "hopper-random-v2":             (1e-2, 0.2, 2, 1e-2, 1),
    "hopper-medium-v2":             (1e-2, 0.0, 10, 1e-1, 1),
    "hopper-medium-replay-v2":      (1e-2, 0.0, 10, 1e-1, 1),
    "hopper-medium-expert-v2":      (1e-4, 0.2, 10, None, None),
    "walker2d-random-v2":           (1e-2, 0.0, 10, 1e-4, 10),
    "walker2d-medium-v2":           (1e-2, 0.0, 10, 1e-1, 1),
    "walker2d-medium-replay-v2":    (1e-2, 0.0, 10, 1e-2, 1),
    "walker2d-medium-expert-v2":    (1e-4, 0.2, 10, None, None),
}


# Paper Table 10: AntMaze.
# reward_scale, offline_beta, offline_alpha, online_beta
ANTMAZE_TABLE = {
    "antmaze-umaze-v2":          (10_000.0, 0.005, 0.001, 0.0),
    "antmaze-umaze-diverse-v2":  (10_000.0, 0.005, 0.001, 0.001),
    "antmaze-medium-play-v2":     (1_000.0, 0.01, 0.001, 0.0),
    "antmaze-medium-diverse-v2":  (1_000.0, 0.01, 0.001, 0.0),
    "antmaze-large-play-v2":      (1_000.0, 0.01, 0.001, 0.01),
    "antmaze-large-diverse-v2":   (10_000.0, 0.01, 0.01, 0.01),
}


# Paper Table 10: Adroit offline hyperparameters.
# beta, alpha.
ADROIT_OFFLINE = {
    "pen-cloned-v1":       (0.01, 0.01),
    "door-cloned-v1":      (0.01, 0.01),
    "hammer-cloned-v1":    (0.1, 0.001),
    "relocate-cloned-v1":  (0.01, 0.01),
    "pen-expert-v1":       (0.01, 0.01),
    "door-expert-v1":      (0.1, 0.001),
    "hammer-expert-v1":    (0.01, 0.001),
    "relocate-expert-v1":  (0.1, 0.001),
}

# Official online beta for cloned tasks.
ADROIT_ONLINE_BETA = {
    "pen": 0.0,
    "door": 0.01,
    "hammer": 0.0,
    "relocate": 0.01,
}


def _domain(env_name: str) -> str:
    name = env_name.lower()
    if "antmaze" in name:
        return "antmaze"
    for token in ("halfcheetah", "hopper", "walker2d", "pen", "door", "hammer", "relocate"):
        if token in name:
            return token
    raise ValueError(f"Unsupported environment: {env_name}")


def _adroit_task(env_name: str) -> str:
    for task in ("pen", "door", "hammer", "relocate"):
        if env_name.startswith(task + "-"):
            return task
    raise ValueError(env_name)


def _transfer_medium_expert_online(env_name: str) -> Tuple[float, int]:
    """Fixed same-domain transfer for MuJoCo medium-expert.

    The PARS paper reports offline results for medium-expert but no online
    hyperparameters. We transfer the corresponding medium task's online alpha
    and actor critic-sample count, without task-specific tuning.
    """
    base = env_name.replace("medium-expert", "medium")
    values = MUJOCO_TABLE[base]
    return float(values[3]), int(values[4])


def _apply_env_config(config: TrainConfig) -> None:
    if config.env_name not in SUPPORTED_ENVS:
        raise ValueError(
            f"{config.env_name} is not in the supported 30-task RLPD-matched benchmark."
        )

    official = config.env_name in OFFICIAL_ONLINE_ENVS
    config.config_source = "official" if official else "adapted"

    if config.strict_official_only and not official:
        raise ValueError(
            f"{config.env_name} was not evaluated in the original PARS online "
            "fine-tuning benchmark. Disable strict_official_only to use the "
            "fixed same-domain adaptation."
        )

    if not config.auto_env_config:
        config.config_source += ":manual_override"
        return

    name = config.env_name
    domain = _domain(name)

    # Paper Table 9 general settings.
    config.hidden_dim = 256
    config.layer_norm = True
    config.critic_learning_rate = 3e-4
    config.actor_learning_rate = 3e-4
    config.online_actor_learning_rate = 3e-4
    config.tau = 5e-3
    config.batch_size = 256
    config.utd_ratio = 20
    config.learning_starts = 0
    config.online_ood_action_weight = 100.0

    if domain == "antmaze":
        reward_scale, beta, alpha, online_beta = ANTMAZE_TABLE[name]
        config.discount = 0.995
        config.reward_scale = reward_scale
        config.q_min = 0.0
        config.offline_beta = beta
        config.offline_alpha = alpha
        config.online_beta = online_beta
        config.online_alpha = 0.001
        config.offline_ood_action_weight = 1000.0
        config.exploration_noise = 0.05
        config.offline_ratio = 0.5
        config.actor_cosine_scheduler = False

        # Released AntMaze YAMLs use a smaller ensemble and deeper critic.
        config.num_ensemble = 4
        config.num_sample_ensemble = 2
        config.online_num_sample_ensemble = 2
        config.online_num_action_sample_ensemble = 1
        config.critic_hidden_layers = 3
        config.policy_noise = 0.0
        config.online_policy_noise = 0.0

    elif domain in ("halfcheetah", "hopper", "walker2d"):
        off_alpha, policy_noise, sk_critic, on_alpha, sk_actor = MUJOCO_TABLE[name]

        config.discount = 0.99
        config.reward_scale = 5.0 if domain == "halfcheetah" else 10.0
        config.q_min = Q_MIN_BY_DOMAIN[domain]
        config.offline_beta = 0.0
        config.offline_alpha = float(off_alpha)
        config.policy_noise = float(policy_noise)
        config.online_policy_noise = 0.0
        config.num_ensemble = 10
        config.num_sample_ensemble = int(sk_critic)
        config.online_num_sample_ensemble = 2
        config.critic_hidden_layers = 2
        config.actor_cosine_scheduler = False
        config.exploration_noise = 0.1
        config.offline_ood_action_weight = 100.0

        # Paper: 5% offline data for HalfCheetah and random datasets,
        # 50% for the remaining MuJoCo online tasks.
        config.offline_ratio = (
            0.05 if (domain == "halfcheetah" or "-random-" in name) else 0.5
        )

        if on_alpha is None:
            # medium-expert is an online-benchmark adaptation.
            adapted_alpha, adapted_actor_samples = _transfer_medium_expert_online(name)
            config.online_alpha = adapted_alpha
            config.online_num_action_sample_ensemble = adapted_actor_samples
        else:
            config.online_alpha = float(on_alpha)
            config.online_num_action_sample_ensemble = int(sk_actor)

        config.online_beta = 0.0

    else:
        # Adroit.
        task = _adroit_task(name)
        config.discount = 0.99
        config.reward_scale = 10.0
        config.q_min = Q_MIN_BY_DOMAIN[task]
        config.offline_ood_action_weight = 100.0
        config.exploration_noise = 0.05
        config.offline_ratio = 0.5
        config.online_alpha = 0.001
        config.online_beta = ADROIT_ONLINE_BETA[task]
        config.num_ensemble = 10
        config.num_sample_ensemble = 2
        config.online_num_sample_ensemble = 2
        config.online_num_action_sample_ensemble = 1
        config.critic_hidden_layers = 2
        config.policy_noise = 0.0
        config.online_policy_noise = 0.0
        config.actor_cosine_scheduler = True

        if name in ADROIT_OFFLINE:
            beta, alpha = ADROIT_OFFLINE[name]
        else:
            # Human datasets are absent from the PARS paper. Use the cloned
            # hyperparameters for the same task as a fixed transfer rule.
            beta, alpha = ADROIT_OFFLINE[f"{task}-cloned-v1"]
        config.offline_beta = float(beta)
        config.offline_alpha = float(alpha)

    if config.scheduler_steps is None:
        config.scheduler_steps = config.pretrain_steps


def _validate_config(config: TrainConfig) -> None:
    _apply_env_config(config)

    for key in (
        "pretrain_steps",
        "num_total_steps",
        "eval_every",
        "eval_episodes",
        "batch_size",
        "num_ensemble",
        "num_sample_ensemble",
        "online_num_sample_ensemble",
        "online_num_action_sample_ensemble",
        "utd_ratio",
    ):
        if int(getattr(config, key)) <= 0:
            raise ValueError(f"{key} must be positive.")

    if config.num_sample_ensemble > config.num_ensemble:
        raise ValueError("num_sample_ensemble cannot exceed num_ensemble.")
    if config.online_num_sample_ensemble > config.num_ensemble:
        raise ValueError("online_num_sample_ensemble cannot exceed num_ensemble.")
    if config.online_num_action_sample_ensemble > config.num_ensemble:
        raise ValueError("online_num_action_sample_ensemble cannot exceed num_ensemble.")
    if not 0.0 <= config.offline_ratio <= 1.0:
        raise ValueError("offline_ratio must be in [0,1].")
    if config.reward_scale <= 0:
        raise ValueError("reward_scale must be positive.")


# =============================================================================
# Environment helpers
# =============================================================================


def _set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
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


def _reset_env(env: gym.Env, seed: Optional[int] = None):
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None:
            _seed_env(env, seed)
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step_env(env: gym.Env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return next_state, float(reward), done, bool(terminated), bool(truncated), dict(info)
    next_state, reward, done, info = result
    info = dict(info)
    truncated = bool(info.get("TimeLimit.truncated", False))
    terminated = bool(done and not truncated)
    return next_state, float(reward), bool(done), terminated, truncated, info


def _max_episode_steps(env: gym.Env) -> int:
    candidate = env
    for _ in range(16):
        value = getattr(candidate, "_max_episode_steps", None)
        if value is not None:
            return int(value)
        spec = getattr(candidate, "spec", None)
        if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
            return int(spec.max_episode_steps)
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    return 1000


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


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def observation(self, observation):
        return ((np.asarray(observation, dtype=np.float32) - self.mean) / self.std).astype(
            np.float32
        )


# =============================================================================
# Replay buffers
# =============================================================================


class OfflineReplayBuffer:
    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        reward_scale: float,
        seed: int,
        normalize_states: bool,
    ):
        self.observations = np.asarray(dataset["observations"], dtype=np.float32).copy()
        self.actions = np.asarray(dataset["actions"], dtype=np.float32).copy()
        self.next_observations = np.asarray(
            dataset["next_observations"], dtype=np.float32
        ).copy()
        self.rewards = (
            np.asarray(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
            * float(reward_scale)
        )
        terminals = np.asarray(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        self.not_dones = 1.0 - terminals
        self.size = len(self.observations)
        self.rng = np.random.default_rng(seed)

        if normalize_states:
            self.state_mean = self.observations.mean(axis=0).astype(np.float32)
            self.state_std = (self.observations.std(axis=0) + 1e-3).astype(np.float32)
            self.observations = (
                (self.observations - self.state_mean) / self.state_std
            ).astype(np.float32)
            self.next_observations = (
                (self.next_observations - self.state_mean) / self.state_std
            ).astype(np.float32)
        else:
            self.state_mean = np.zeros(self.observations.shape[1:], dtype=np.float32)
            self.state_std = np.ones(self.observations.shape[1:], dtype=np.float32)

    def sample_numpy(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx],
            self.not_dones[idx],
        )


class OnlineReplayBuffer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.pointer = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        reward: float,
        terminated: bool,
    ) -> None:
        i = self.pointer
        self.observations[i] = (
            np.asarray(observation, dtype=np.float32) - self.state_mean
        ) / self.state_std
        self.actions[i] = np.asarray(action, dtype=np.float32)
        self.next_observations[i] = (
            np.asarray(next_observation, dtype=np.float32) - self.state_mean
        ) / self.state_std
        self.rewards[i] = float(reward)
        self.not_dones[i] = 1.0 - float(terminated)
        self.pointer = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_numpy(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("Online replay buffer is empty.")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx],
            self.not_dones[idx],
        )


def _to_torch(batch: Tuple[np.ndarray, ...], device: torch.device):
    return tuple(
        torch.as_tensor(x, dtype=torch.float32, device=device)
        for x in batch
    )


def _mixed_batch(
    offline: OfflineReplayBuffer,
    online: OnlineReplayBuffer,
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
):
    offline_n = int(batch_size * offline_ratio)
    offline_n = min(max(offline_n, 0), batch_size)
    online_n = batch_size - offline_n

    parts = []
    if offline_n > 0:
        parts.append(offline.sample_numpy(offline_n))
    if online_n > 0:
        parts.append(online.sample_numpy(online_n))

    if len(parts) == 1:
        return _to_torch(parts[0], device)

    merged = tuple(
        np.concatenate([p[field] for p in parts], axis=0)
        for field in range(5)
    )
    return _to_torch(merged, device)


# =============================================================================
# Networks
# =============================================================================


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size: int, in_dim: int, out_dim: int, final=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_dim))
        if final:
            nn.init.uniform_(self.weight, -3e-3, 3e-3)
            nn.init.uniform_(self.bias, -3e-3, 3e-3)
        else:
            # Approximate Flax he_uniform for each ensemble member.
            for e in range(ensemble_size):
                nn.init.kaiming_uniform_(self.weight[e], a=math.sqrt(5), nonlinearity="relu")
            nn.init.constant_(self.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,I] -> [E,B,O], or x [E,B,I] -> [E,B,O].
        if x.ndim == 2:
            return torch.einsum("bi,eoi->ebo", x, self.weight) + self.bias
        if x.ndim == 3:
            return torch.einsum("ebi,eoi->ebo", x, self.weight) + self.bias
        raise ValueError(f"Unexpected EnsembleLinear input shape: {tuple(x.shape)}")


class EnsembleLayerNorm(nn.Module):
    def __init__(self, ensemble_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, 1, dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None)
        return x * self.weight + self.bias


class EnsembleCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        ensemble_size: int,
        num_hidden_layers: int,
        layer_norm: bool,
    ):
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_dim = state_dim + action_dim
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                EnsembleLinear(self.ensemble_size, in_dim, hidden_dim, final=False)
            )
            if layer_norm:
                self.norm_layers.append(EnsembleLayerNorm(self.ensemble_size, hidden_dim))
            in_dim = hidden_dim
        self.use_layer_norm = bool(layer_norm)
        self.output = EnsembleLinear(self.ensemble_size, in_dim, 1, final=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = F.relu(x)
        return self.output(x).squeeze(-1)  # [E,B]


# =============================================================================
# PARS learner
# =============================================================================


class PARSLearner:
    def __init__(
        self,
        config: TrainConfig,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.max_action = float(max_action)
        self.action_dim = int(action_dim)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor).to(device)
        self.critic = EnsembleCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            ensemble_size=config.num_ensemble,
            num_hidden_layers=config.critic_hidden_layers,
            layer_norm=config.layer_norm,
        ).to(device)
        self.critic_target = deepcopy(self.critic).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

        self.actor_scheduler = None
        if config.actor_cosine_scheduler:
            t_max = max(
                1,
                int(config.scheduler_steps or config.pretrain_steps)
                // max(1, config.policy_frequency),
            )
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=t_max,
                eta_min=config.min_actor_learning_rate,
            )

        self.total_it = 0
        self.online_it = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        return self.actor(tensor).squeeze(0).cpu().numpy()

    def _sample_critic_indices(self, count: int) -> torch.Tensor:
        if count >= self.config.num_ensemble:
            return torch.arange(self.config.num_ensemble, device=self.device)
        return torch.randperm(self.config.num_ensemble, device=self.device)[:count]

    def _target_action(
        self, next_state: torch.Tensor, action_shape: torch.Size, online: bool
    ) -> torch.Tensor:
        noise_std = (
            self.config.online_policy_noise if online else self.config.policy_noise
        )
        noise = (
            torch.randn(action_shape, device=self.device) * noise_std
        ).clamp(-self.config.noise_clip, self.config.noise_clip)
        return (
            self.actor_target(next_state) + noise
        ).clamp(-self.max_action, self.max_action)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - tau).add_(sp, alpha=tau)

    def _critic_loss(
        self,
        batch,
        online: bool,
    ):
        state, action, next_state, reward, not_done = batch
        alpha = self.config.online_alpha if online else self.config.offline_alpha
        sample_count = (
            self.config.online_num_sample_ensemble
            if online
            else self.config.num_sample_ensemble
        )

        with torch.no_grad():
            next_action = self._target_action(next_state, action.shape, online=online)
            target_q_all = self.critic_target(next_state, next_action)
            indices = self._sample_critic_indices(sample_count)
            target_q = target_q_all.index_select(0, indices).min(dim=0).values.unsqueeze(-1)
            td_target = reward + not_done * self.config.discount * target_q

        current_q = self.critic(state, action)
        bellman_loss = F.mse_loss(
            current_q, td_target.squeeze(-1).unsqueeze(0).expand_as(current_q)
        )

        if alpha > 0.0:
            base = torch.rand_like(action) * 2.0 - 1.0
            infeasible = torch.where(base < 0.0, base - 1.0, base + 1.0)
            distance = (
                self.config.online_ood_action_weight
                if online
                else self.config.offline_ood_action_weight
            )
            infeasible = infeasible * float(distance)
            q_ood = self.critic(state, infeasible)
            q_min_target = torch.full_like(q_ood, float(self.config.q_min))
            ood_loss = F.mse_loss(q_ood, q_min_target)
        else:
            q_ood = torch.zeros_like(current_q)
            ood_loss = torch.zeros((), device=self.device)

        total = bellman_loss + float(alpha) * ood_loss
        metrics = {
            "critic_loss": float(total.detach().item()),
            "q_loss": float(bellman_loss.detach().item()),
            "q_ood_loss": float(ood_loss.detach().item()),
            "q": float(current_q[0].detach().mean().item()),
            "q_ood": float(q_ood[0].detach().mean().item()),
        }
        return total, metrics

    def _actor_loss(self, state: torch.Tensor, bc_action: torch.Tensor, online: bool):
        action = self.actor(state)
        q_all = self.critic(state, action)

        if online:
            indices = self._sample_critic_indices(
                self.config.online_num_action_sample_ensemble
            )
            q_value = q_all.index_select(0, indices).mean(dim=0)
            beta = float(self.config.online_beta)
        else:
            q_value = q_all.min(dim=0).values
            beta = float(self.config.offline_beta)

        bc_penalty = ((action - bc_action) ** 2).sum(dim=-1)
        if beta == 0.0 and online:
            scale = torch.ones((), device=self.device)
        else:
            scale = 1.0 / (q_value.abs().mean().detach() + 1e-10)

        loss = (beta * bc_penalty - scale * q_value).mean()
        return loss, {
            "actor_loss": float(loss.detach().item()),
            "actor_q": float(q_value.detach().mean().item()),
            "bc_penalty": float(bc_penalty.detach().mean().item()),
            "actor_q_scale": float(scale.detach().item()),
        }

    def offline_update(self, batch) -> Dict[str, float]:
        self.total_it += 1

        critic_loss, metrics = self._critic_loss(batch, online=False)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.config.policy_frequency == 0:
            state, action, _, _, _ = batch
            # Freeze critic parameters while retaining gradient through Q wrt action.
            critic_requires_grad = [p.requires_grad for p in self.critic.parameters()]
            for p in self.critic.parameters():
                p.requires_grad_(False)
            actor_loss, actor_metrics = self._actor_loss(state, action, online=False)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for p, old in zip(self.critic.parameters(), critic_requires_grad):
                p.requires_grad_(old)

            if self.actor_scheduler is not None:
                self.actor_scheduler.step()

            self._soft_update(self.actor_target, self.actor, self.config.tau)
            self._soft_update(self.critic_target, self.critic, self.config.tau)
            metrics.update(actor_metrics)

        return {f"offline/{k}": v for k, v in metrics.items()}

    def start_online_phase(self) -> None:
        # Official code resets actor optimizer and changes policy frequency to 1.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.online_actor_learning_rate
        )
        self.actor_scheduler = None
        self.online_it = 0

    def online_update(
        self,
        offline_buffer: OfflineReplayBuffer,
        online_buffer: OnlineReplayBuffer,
    ) -> Dict[str, float]:
        self.total_it += 1
        self.online_it += 1
        last_batch = None
        aggregate = {}

        # Match released code: UTD critic steps per environment step.
        for _ in range(self.config.utd_ratio):
            batch = _mixed_batch(
                offline_buffer,
                online_buffer,
                self.config.batch_size,
                self.config.offline_ratio,
                self.device,
            )
            last_batch = batch
            critic_loss, metrics = self._critic_loss(batch, online=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()
            aggregate = metrics

        if last_batch is None:
            return {}

        # Released code sets policy_freq=1 for online training, hence one actor
        # update and one target update per environment step after UTD critic steps.
        state, action, _, _, _ = last_batch
        old_flags = [p.requires_grad for p in self.critic.parameters()]
        for p in self.critic.parameters():
            p.requires_grad_(False)
        actor_loss, actor_metrics = self._actor_loss(state, action, online=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for p, old in zip(self.critic.parameters(), old_flags):
            p.requires_grad_(old)

        self._soft_update(self.actor_target, self.actor, self.config.tau)
        self._soft_update(self.critic_target, self.critic, self.config.tau)

        aggregate.update(actor_metrics)
        aggregate["offline_ratio"] = float(self.config.offline_ratio)
        aggregate["utd_ratio"] = float(self.config.utd_ratio)
        return {f"online/{k}": v for k, v in aggregate.items()}

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "online_it": self.online_it,
        }


# =============================================================================
# Evaluation / logging
# =============================================================================


@torch.no_grad()
def evaluate(
    learner: PARSLearner,
    env: gym.Env,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    learner.actor.eval()
    returns = []
    lengths = []

    for ep in range(episodes):
        state = _reset_env(env, seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            normalized = (
                np.asarray(state, dtype=np.float32) - state_mean
            ) / state_std
            action = learner.select_action(normalized)
            state, reward, done, _, _, _ = _step_env(env, action)
            ep_return += reward
            ep_length += 1
        returns.append(ep_return)
        lengths.append(ep_length)

    learner.actor.train()
    raw_return = float(np.mean(returns))
    result = {
        "return": raw_return,
        "length": float(np.mean(lengths)),
    }
    normalized_score = _get_normalized_score(env, raw_return)
    if normalized_score is not None:
        result["d4rl_normalized_score"] = normalized_score
    return result


def _write_metrics(
    writer: SummaryWriter,
    metrics: Mapping[str, float],
    step: int,
) -> None:
    for key, value in metrics.items():
        if math.isfinite(float(value)):
            writer.add_scalar(key, float(value), step)


# =============================================================================
# Training
# =============================================================================


def train(config: TrainConfig) -> None:
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)
    device = torch.device(config.device)

    base_train_env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)
    _seed_env(base_train_env, config.train_seed)
    _seed_env(eval_env, config.eval_seed)

    if not isinstance(base_train_env.observation_space, gym.spaces.Box):
        raise TypeError("PARS requires a continuous Box observation space.")
    if not isinstance(base_train_env.action_space, gym.spaces.Box):
        raise TypeError("PARS requires a continuous Box action space.")

    dataset = d4rl.qlearning_dataset(base_train_env)
    offline_buffer = OfflineReplayBuffer(
        dataset=dataset,
        reward_scale=config.reward_scale,
        seed=config.train_seed + 1,
        normalize_states=config.normalize_states,
    )

    state_mean = offline_buffer.state_mean
    state_std = offline_buffer.state_std
    train_env = base_train_env

    state_dim = int(np.prod(base_train_env.observation_space.shape))
    action_dim = int(np.prod(base_train_env.action_space.shape))
    max_action = float(np.asarray(base_train_env.action_space.high).reshape(-1)[0])

    online_buffer = OnlineReplayBuffer(
        observation_dim=state_dim,
        action_dim=action_dim,
        capacity=config.replay_buffer_size,
        seed=config.train_seed + 2,
        state_mean=state_mean,
        state_std=state_std,
    )

    learner = PARSLearner(
        config=config,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
    )

    run_name = f"{config.env_name}_seed{config.train_seed}_PARS"
    log_dir = os.path.join(
        config.log_root,
        config.env_name.split("-")[0],
        run_name,
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"env={config.env_name}\n"
        f"config_source={config.config_source}\n"
        f"device={device}\n"
        f"offline_dataset_size={offline_buffer.size}\n"
        f"pretrain_steps={config.pretrain_steps}\n"
        f"online_steps={config.num_total_steps}\n"
        f"reward_scale={config.reward_scale}\n"
        f"q_min={config.q_min}\n"
        f"offline_alpha={config.offline_alpha}, offline_beta={config.offline_beta}\n"
        f"online_alpha={config.online_alpha}, online_beta={config.online_beta}\n"
        f"ensemble={config.num_ensemble}, "
        f"offline_target_samples={config.num_sample_ensemble}, "
        f"online_target_samples={config.online_num_sample_ensemble}, "
        f"online_actor_samples={config.online_num_action_sample_ensemble}\n"
        f"offline_ratio={config.offline_ratio}, UTD={config.utd_ratio}\n"
        f"log_dir={log_dir}",
        flush=True,
    )

    if config.config_source.startswith("adapted"):
        print(
            "[PARS] NOTE: This task was not evaluated in the original PARS "
            "online-finetuning benchmark. A fixed same-domain hyperparameter "
            "transfer is being used; no task-specific tuning is performed.",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Offline pre-training.
    # -------------------------------------------------------------------------
    for step in trange(config.pretrain_steps, desc="PARS offline pre-training"):
        batch = _to_torch(
            offline_buffer.sample_numpy(config.batch_size),
            device,
        )
        metrics = learner.offline_update(batch)

        if (step + 1) % config.log_every == 0:
            _write_metrics(writer, metrics, step + 1)

        if (step + 1) % config.eval_every == 0:
            eval_metrics = evaluate(
                learner,
                eval_env,
                state_mean,
                state_std,
                config.eval_episodes,
                config.eval_seed,
            )
            tagged = {f"offline_evaluation/{k}": v for k, v in eval_metrics.items()}
            _write_metrics(writer, tagged, step + 1)
            print(
                f"[offline eval] step={step + 1} "
                f"return={eval_metrics['return']:.3f} "
                + (
                    f"score={eval_metrics['d4rl_normalized_score']:.3f}"
                    if "d4rl_normalized_score" in eval_metrics
                    else ""
                ),
                flush=True,
            )
            writer.flush()

    # -------------------------------------------------------------------------
    # Online fine-tuning.
    # -------------------------------------------------------------------------
    learner.start_online_phase()
    state = _reset_env(train_env)
    episode_return = 0.0
    episode_steps = 0
    max_episode_steps = _max_episode_steps(train_env)

    for t in trange(config.num_total_steps, desc="PARS online fine-tuning"):
        normalized_state = (
            np.asarray(state, dtype=np.float32) - state_mean
        ) / state_std

        if t < config.learning_starts:
            action = train_env.action_space.sample()
        else:
            action = learner.select_action(normalized_state)
            noise = np.random.normal(
                0.0,
                max_action * config.exploration_noise,
                size=action_dim,
            ).astype(np.float32)
            action = np.clip(action + noise, -max_action, max_action)

        next_state, raw_reward, done, terminated, truncated, _ = _step_env(
            train_env, action
        )
        episode_steps += 1

        # Match original code: a time-limit ending is not an MDP terminal.
        timeout = bool(truncated or (done and episode_steps >= max_episode_steps))
        true_terminal = bool(terminated and not timeout)

        training_reward = float(raw_reward) * float(config.reward_scale)
        online_buffer.add(
            state,
            action,
            next_state,
            training_reward,
            true_terminal,
        )

        episode_return += raw_reward
        state = next_state

        if t >= config.learning_starts:
            metrics = learner.online_update(offline_buffer, online_buffer)
            if (t + 1) % config.log_every == 0:
                _write_metrics(writer, metrics, t + 1)

        if done:
            writer.add_scalar("online_training/return", episode_return, t + 1)
            writer.add_scalar("online_training/length", episode_steps, t + 1)
            state = _reset_env(train_env)
            episode_return = 0.0
            episode_steps = 0

        if (t + 1) % config.eval_every == 0:
            eval_metrics = evaluate(
                learner,
                eval_env,
                state_mean,
                state_std,
                config.eval_episodes,
                config.eval_seed,
            )
            tagged = {f"online_evaluation/{k}": v for k, v in eval_metrics.items()}
            _write_metrics(writer, tagged, t + 1)
            print(
                f"[online eval] step={t + 1} "
                f"return={eval_metrics['return']:.3f} "
                + (
                    f"score={eval_metrics['d4rl_normalized_score']:.3f}"
                    if "d4rl_normalized_score" in eval_metrics
                    else ""
                ),
                flush=True,
            )
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
            },
            os.path.join(checkpoint_dir, "final.pt"),
        )

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train(pyrallis.parse(config_class=TrainConfig))
