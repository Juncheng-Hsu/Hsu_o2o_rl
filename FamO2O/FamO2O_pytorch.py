"""Single-file PyTorch implementation of FamO2O over IQL.

This file ports the official JAX IQL implementation of:

    Train Once, Get a Family: State-Adaptive Balances for
    Offline-to-Online Reinforcement Learning (NeurIPS 2023).

The implementation keeps the user's existing single-file PyTorch style while
following both the paper and the author-released JAX code:

1. The base algorithm is Implicit Q-Learning (IQL), which is the main FamO2O
   implementation and the version used for the paper's principal D4RL results.
2. A universal policy pi_u(a | s, beta) is conditioned on a sinusoidal encoding
   of the balance coefficient beta.
3. A stochastic balance policy pi_b(beta | s) selects a state-adaptive balance
   coefficient.
4. During offline pre-training, beta is sampled uniformly from the configured
   range for each transition. During online fine-tuning, beta is sampled from
   the balance policy for policy regression.
5. The universal-policy objective is the IQL advantage-weighted regression
   objective exp(beta * (Q - V)) * log pi_u, capped at 100 as in the official
   implementation.
6. The balance policy is optimized through the universal policy to maximize the
   target-critic value, with an automatically tuned entropy temperature.
7. The replay buffer is initialized with the complete offline dataset. Online
   transitions are appended to the same buffer, and training batches are drawn
   uniformly from the resulting combined replay, rather than with a fixed
   offline/online ratio.
8. The update order matches the released FamilyLearner: V, balance policy,
   balance temperature, universal policy, Q, then target-Q soft update. For the
   online universal-policy update, coefficients are sampled from the balance
   policy before that iteration's balance-policy optimizer step, matching the
   functional JAX update semantics.

The default settings correspond to the official IQL+FamO2O AntMaze setup,
using two 256-unit hidden layers, expectile 0.9, coefficient range [8, 14],
batch size 256, replay capacity 2,000,000, one million offline gradient steps,
and one million online environment/gradient steps. The default environment name
uses the modern D4RL v2 suffix; the released README used the earlier v0 names.

PyTorch and JAX/Flax have different PRNG streams, parameter layouts, and low-
level kernels, so identical seeds do not imply bitwise-identical parameters or
trajectories. This is an algorithmically faithful PyTorch port, not a numerical
bit-for-bit reproduction.
"""

import math
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

# Avoid irrelevant optional-domain import failures in older D4RL installations.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import d4rl  # noqa: F401: importing d4rl registers Gym environments
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
    # The official README used AntMaze v0. v2 is the commonly available D4RL
    # equivalent in current installations and matches the user's experiments.
    env_name: str = "antmaze-umaze-v2"

    # Official IQL+FamO2O network configuration.
    hidden_dim: int = 256
    hidden_layers: int = 2

    actor_learning_rate: float = 3e-4
    balance_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4

    discount: float = 0.99
    tau: float = 5e-3

    # Set these to None to select the published domain-specific setting:
    # AntMaze: expectile=0.9, beta in [8,14], balance update every step.
    # Locomotion: expectile=0.7, beta in [1,5], balance update every 5 steps.
    expectile: Optional[float] = None
    family_coefficient_min: Optional[float] = None
    family_coefficient_max: Optional[float] = None
    balance_update_every: Optional[int] = None

    family_sin_cos_n: float = 10_000.0
    family_sin_cos_d: int = 6
    max_advantage_weight: float = 100.0

    universal_log_std_min: float = -5.0
    universal_log_std_max: float = 2.0
    balance_log_std_min: float = -10.0
    balance_log_std_max: float = 2.0

    init_temperature: float = 1.0
    # The official learner defaults to -1/2 because the balance action is scalar.
    target_entropy: Optional[float] = None

    batch_size: int = 256
    replay_buffer_size: int = 2_000_000
    init_dataset_size: Optional[int] = None
    pretrain_steps: int = 1_000_000
    num_total_steps: int = 1_000_000

    # Official README evaluation protocol for the released AntMaze code.
    eval_episodes: int = 100
    eval_every: int = 100_000
    last_eval_every: int = 10_000
    last_eval_start_steps: int = 60_000
    log_every: int = 1_000

    train_seed: int = 42
    eval_seed_offset: int = 42
    deterministic_torch: bool = False

    checkpoints_path: Optional[str] = None
    log_root: str = "logs/FamO2O_PyTorch"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


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
    """PyTorch equivalent of the official SinglePrecision wrapper."""

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
# D4RL dataset and the single combined replay buffer used by FamO2O
# -----------------------------------------------------------------------------


class D4RLDataset:
    def __init__(
        self,
        env: gym.Env,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        dataset = d4rl.qlearning_dataset(env)
        dataset = {
            key: np.asarray(value).copy()
            for key, value in dataset.items()
        }

        if clip_to_eps:
            limit = 1.0 - eps
            dataset["actions"] = np.clip(
                dataset["actions"], -limit, limit
            )

        dones_float = np.zeros_like(dataset["rewards"], dtype=np.float32)
        for index in range(len(dones_float) - 1):
            discontinuity = (
                np.linalg.norm(
                    dataset["observations"][index + 1]
                    - dataset["next_observations"][index]
                )
                > 1e-6
            )
            if discontinuity or dataset["terminals"][index] == 1.0:
                dones_float[index] = 1.0
        dones_float[-1] = 1.0

        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(
            np.float32
        )
        self.size = len(self.observations)


class ReplayBuffer:
    """One replay buffer containing the offline dataset and online additions."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        seed: Optional[int] = None,
    ):
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("ReplayBuffer requires a flat Box observation space.")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("ReplayBuffer requires a Box action space.")
        if capacity < 1:
            raise ValueError("Replay-buffer capacity must be positive.")

        self.capacity = int(capacity)
        self.size = 0
        self.insert_index = 0
        self.observations = np.empty(
            (capacity, *observation_space.shape), dtype=np.float32
        )
        self.actions = np.empty(
            (capacity, *action_space.shape), dtype=np.float32
        )
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.dones_float = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self._np_random, self.seed_value = gym.utils.seeding.np_random(seed)

    def initialize_with_dataset(
        self,
        dataset: D4RLDataset,
        num_samples: Optional[int] = None,
    ) -> None:
        if self.insert_index != 0 or self.size != 0:
            raise RuntimeError(
                "The offline dataset can only initialize an empty replay buffer."
            )

        dataset_size = dataset.size
        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(int(num_samples), dataset_size)

        if num_samples > self.capacity:
            raise ValueError(
                f"Offline dataset size {num_samples} exceeds replay capacity "
                f"{self.capacity}."
            )

        if num_samples < dataset_size:
            if hasattr(self._np_random, "permutation"):
                indices = self._np_random.permutation(dataset_size)[:num_samples]
            else:
                indices = np.random.permutation(dataset_size)[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]
        self.insert_index = num_samples % self.capacity
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ) -> None:
        index = self.insert_index
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.masks[index] = mask
        self.dones_float[index] = done_float
        self.next_observations[index] = next_observation
        self.insert_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        if hasattr(self._np_random, "integers"):
            indices = self._np_random.integers(self.size, size=batch_size)
        else:
            indices = self._np_random.randint(self.size, size=batch_size)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "next_observations": self.next_observations[indices],
        }


# -----------------------------------------------------------------------------
# Reward preprocessing from the official fine-tuning script
# -----------------------------------------------------------------------------


def _is_locomotion_env(env_name: str) -> bool:
    name = env_name.lower()
    return any(token in name for token in ("halfcheetah", "hopper", "walker2d"))


def _is_antmaze_env(env_name: str) -> bool:
    return "antmaze" in env_name.lower()


def _compute_locomotion_return_range(dataset: D4RLDataset) -> Tuple[float, float]:
    returns = []
    episode_return = 0.0
    for reward, done_float in zip(dataset.rewards, dataset.dones_float):
        episode_return += float(reward)
        if done_float == 1.0:
            returns.append(episode_return)
            episode_return = 0.0

    if not returns:
        raise RuntimeError("No complete trajectories found in the D4RL dataset.")

    return_min = float(np.min(returns))
    return_max = float(np.max(returns))
    if not return_max > return_min:
        raise RuntimeError(
            "Locomotion reward normalization requires return_max > return_min."
        )
    return return_min, return_max


def _transform_batch_rewards(
    batch: NumpyBatch,
    env_name: str,
    locomotion_return_range: Optional[Tuple[float, float]],
) -> NumpyBatch:
    transformed = dict(batch)
    rewards = np.asarray(batch["rewards"], dtype=np.float32)

    if _is_antmaze_env(env_name):
        rewards = rewards - 1.0
    elif _is_locomotion_env(env_name):
        if locomotion_return_range is None:
            raise RuntimeError("Missing locomotion return-normalization range.")
        return_min, return_max = locomotion_return_range
        rewards = rewards / (return_max - return_min) * 1000.0

    transformed["rewards"] = rewards.astype(np.float32)
    return transformed


# -----------------------------------------------------------------------------
# Flax-equivalent orthogonal MLPs and FamO2O policies
# -----------------------------------------------------------------------------


def _init_linear_orthogonal(
    layer: nn.Linear,
    gain: float = math.sqrt(2.0),
) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Equivalent to the official MLP with orthogonal sqrt(2) initialization."""

    def __init__(
        self,
        input_dim: int,
        layer_dims: Sequence[int],
        activate_final: bool = False,
    ):
        super().__init__()
        if not layer_dims:
            raise ValueError("layer_dims cannot be empty.")

        self.layers = nn.ModuleList()
        current_dim = input_dim
        for output_dim in layer_dims:
            layer = nn.Linear(current_dim, output_dim)
            _init_linear_orthogonal(layer)
            self.layers.append(layer)
            current_dim = output_dim
        self.activate_final = bool(activate_final)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index + 1 < len(self.layers) or self.activate_final:
                values = F.relu(values)
        return values


def sinusoidal_balance_encoding(
    coefficients: torch.Tensor,
    n: float,
    d: int,
) -> torch.Tensor:
    """Exact PyTorch form of the official sin_cos_skill_func."""

    if d < 2 or d % 2 != 0:
        raise ValueError("family_sin_cos_d must be a positive even integer.")
    if n <= 0.0:
        raise ValueError("family_sin_cos_n must be positive.")

    coefficients = coefficients.reshape(-1, 1)
    indices = torch.arange(
        0,
        d // 2,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    denominator = torch.pow(
        torch.as_tensor(n, dtype=coefficients.dtype, device=coefficients.device),
        2.0 * indices / float(d),
    )
    inner = coefficients / denominator.unsqueeze(0)

    encoding = torch.empty(
        (coefficients.shape[0], d),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    encoding[:, 0:d:2] = torch.sin(inner)
    encoding[:, 1:d:2] = torch.cos(inner)
    return encoding


class UniversalPolicy(nn.Module):
    """The coefficient-conditioned IQL policy pi_u(a | s, beta)."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        encoding_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.base = MLP(
            observation_dim + encoding_dim,
            hidden_dims,
            activate_final=True,
        )
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        _init_linear_orthogonal(self.mean)

        # The released implementation uses one learned, state-independent
        # log-standard-deviation vector initialized to zero.
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def distribution(
        self,
        observations_and_encoding: torch.Tensor,
        temperature: float = 1.0,
    ) -> Normal:
        features = self.base(observations_and_encoding)
        mean = torch.tanh(self.mean(features))
        log_std = self.log_std.clamp(
            min=self.log_std_min,
            max=self.log_std_max,
        )
        std = log_std.exp().expand_as(mean)
        if temperature < 0.0:
            raise ValueError("Policy sampling temperature cannot be negative.")
        # Temperature is only used for action sampling. Training distributions
        # call this method with temperature=1, as in the released code.
        std = std * float(temperature)
        return Normal(mean, std)

    def sample(
        self,
        observations_and_encoding: torch.Tensor,
        reparameterize: bool = True,
    ) -> torch.Tensor:
        distribution = self.distribution(observations_and_encoding)
        return (
            distribution.rsample()
            if reparameterize
            else distribution.sample()
        )

    def log_prob(
        self,
        observations_and_encoding: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self.distribution(observations_and_encoding)
        return distribution.log_prob(actions).sum(dim=-1)

    def mode(self, observations_and_encoding: torch.Tensor) -> torch.Tensor:
        features = self.base(observations_and_encoding)
        return torch.tanh(self.mean(features))


class BalancePolicy(nn.Module):
    """The tanh-Gaussian balance policy pi_b(beta_raw | s)."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.base = MLP(
            observation_dim,
            hidden_dims,
            activate_final=True,
        )
        self.mean = nn.Linear(hidden_dims[-1], 1)
        _init_linear_orthogonal(self.mean)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def _normal(self, observations: torch.Tensor) -> Normal:
        features = self.base(observations)
        mean = self.mean(features)
        log_std = self.log_std.clamp(
            min=self.log_std_min,
            max=self.log_std_max,
        )
        return Normal(mean, log_std.exp().expand_as(mean))

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
        distribution = self._normal(observations)
        pre_tanh = (
            distribution.rsample()
            if reparameterize
            else distribution.sample()
        )
        raw_coefficients = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(distribution, pre_tanh)
        return raw_coefficients, log_prob

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        distribution = self._normal(observations)
        return torch.tanh(distribution.mean)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.network = MLP(
            observation_dim + action_dim,
            (*hidden_dims, 1),
            activate_final=False,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        values = self.network(torch.cat([observations, actions], dim=-1))
        return values.squeeze(-1)


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
        return (
            self.q1(observations, actions),
            self.q2(observations, actions),
        )


class ValueNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.network = MLP(
            observation_dim,
            (*hidden_dims, 1),
            activate_final=False,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


# -----------------------------------------------------------------------------
# FamO2O over IQL learner
# -----------------------------------------------------------------------------


@contextmanager
def _frozen_parameters(module: nn.Module):
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), previous):
            parameter.requires_grad_(requires_grad)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(
                source_parameter,
                alpha=tau,
            )


class FamO2OLearner:
    def __init__(
        self,
        universal_policy: UniversalPolicy,
        balance_policy: BalancePolicy,
        critic: DoubleCritic,
        value: ValueNetwork,
        universal_optimizer: torch.optim.Optimizer,
        balance_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        temperature_learning_rate: float,
        init_temperature: float,
        target_entropy: float,
        discount: float,
        tau: float,
        expectile: float,
        coefficient_range: Tuple[float, float],
        family_sin_cos_n: float,
        family_sin_cos_d: int,
        max_advantage_weight: float,
        balance_update_every: int,
        device: torch.device,
    ):
        self.universal_policy = universal_policy
        self.balance_policy = balance_policy
        self.critic = critic
        self.value = value
        self.target_critic = deepcopy(critic).to(device)
        self.target_critic.requires_grad_(False)

        self.universal_optimizer = universal_optimizer
        self.balance_optimizer = balance_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_optimizer = value_optimizer

        self.log_temperature = torch.tensor(
            math.log(init_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature],
            lr=temperature_learning_rate,
        )

        self.target_entropy = float(target_entropy)
        self.discount = float(discount)
        self.tau = float(tau)
        self.expectile = float(expectile)
        self.coefficient_min = float(coefficient_range[0])
        self.coefficient_max = float(coefficient_range[1])
        self.family_sin_cos_n = float(family_sin_cos_n)
        self.family_sin_cos_d = int(family_sin_cos_d)
        self.max_advantage_weight = float(max_advantage_weight)
        self.balance_update_every = int(balance_update_every)
        self.device = device
        self.update_step = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _map_raw_to_coefficient(
        self,
        raw_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (raw_coefficients + 1.0)
            * 0.5
            * (self.coefficient_max - self.coefficient_min)
            + self.coefficient_min
        )

    def _policy_inputs(
        self,
        observations: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        encoding = sinusoidal_balance_encoding(
            coefficients,
            n=self.family_sin_cos_n,
            d=self.family_sin_cos_d,
        )
        return torch.cat([observations, encoding], dim=-1)

    @torch.no_grad()
    def sample_action(
        self,
        observation: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        observations = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)

        # Official inference uses the deterministic mean of the balance model.
        raw_coefficients = self.balance_policy.mode(observations)
        coefficients = self._map_raw_to_coefficient(raw_coefficients)
        policy_inputs = self._policy_inputs(observations, coefficients)

        if temperature == 0.0:
            actions = self.universal_policy.mode(policy_inputs)
        else:
            distribution = self.universal_policy.distribution(
                policy_inputs,
                temperature=temperature,
            )
            actions = distribution.sample()

        return np.clip(
            actions.squeeze(0).cpu().numpy(),
            -1.0,
            1.0,
        )

    def _sample_universal_coefficients(
        self,
        observations: torch.Tensor,
        random_coefficients: bool,
    ) -> torch.Tensor:
        if random_coefficients:
            return torch.empty(
                (observations.shape[0], 1),
                dtype=observations.dtype,
                device=observations.device,
            ).uniform_(self.coefficient_min, self.coefficient_max)

        # This sample must use the pre-update balance policy. The released JAX
        # function passes the old functional model into the universal-policy
        # update even though it returns a newly optimized balance model.
        with torch.no_grad():
            raw_coefficients, _ = self.balance_policy.sample(
                observations,
                reparameterize=False,
            )
            return self._map_raw_to_coefficient(raw_coefficients)

    def update_value(self, batch: TorchBatch) -> Dict[str, float]:
        with torch.no_grad():
            q1, q2 = self.target_critic(
                batch["observations"],
                batch["actions"],
            )
            q = torch.minimum(q1, q2)

        values = self.value(batch["observations"])
        difference = q - values
        weights = torch.where(
            difference > 0.0,
            torch.as_tensor(
                self.expectile,
                dtype=difference.dtype,
                device=difference.device,
            ),
            torch.as_tensor(
                1.0 - self.expectile,
                dtype=difference.dtype,
                device=difference.device,
            ),
        )
        value_loss = (weights * difference.pow(2)).mean()

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        return {
            "value_loss": float(value_loss.detach().item()),
            "v": float(values.detach().mean().item()),
        }

    def update_balance_policy(
        self,
        batch: TorchBatch,
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        observations = batch["observations"]

        # Freeze model parameters but preserve gradients from Q through the
        # universal-policy input back into the sampled balance coefficient.
        with _frozen_parameters(self.universal_policy), _frozen_parameters(
            self.target_critic
        ):
            raw_coefficients, log_prob = self.balance_policy.sample(
                observations,
                reparameterize=True,
            )
            coefficients = self._map_raw_to_coefficient(raw_coefficients)
            policy_inputs = self._policy_inputs(observations, coefficients)
            actions = self.universal_policy.sample(
                policy_inputs,
                reparameterize=True,
            )
            q1, q2 = self.target_critic(observations, actions)
            q = torch.minimum(q1, q2)
            balance_loss = (
                self.temperature.detach() * log_prob - q
            ).mean()
            entropy = -log_prob.mean()

        self.balance_optimizer.zero_grad(set_to_none=True)
        balance_loss.backward()
        self.balance_optimizer.step()

        return (
            {
                "balance_loss": float(balance_loss.detach().item()),
                "balance_entropy": float(entropy.detach().item()),
                "balance_coefficient": float(
                    coefficients.detach().mean().item()
                ),
                "balance_q": float(q.detach().mean().item()),
            },
            entropy.detach(),
        )

    def update_temperature(self, entropy: torch.Tensor) -> Dict[str, float]:
        temperature_before = self.temperature
        temperature_loss = temperature_before * (
            entropy - self.target_entropy
        ).mean()

        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        return {
            "temperature": float(temperature_before.detach().item()),
            "temperature_loss": float(
                temperature_loss.detach().item()
            ),
        }

    def update_universal_policy(
        self,
        batch: TorchBatch,
        coefficients: torch.Tensor,
    ) -> Dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]

        with torch.no_grad():
            values = self.value(observations)
            q1, q2 = self.target_critic(observations, actions)
            q = torch.minimum(q1, q2)
            advantage = q - values
            advantage_weights = torch.exp(
                advantage * coefficients.squeeze(-1)
            ).clamp(max=self.max_advantage_weight)

        policy_inputs = self._policy_inputs(observations, coefficients)
        log_prob = self.universal_policy.log_prob(policy_inputs, actions)
        actor_loss = -(advantage_weights * log_prob).mean()

        self.universal_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.universal_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "advantage": float(advantage.detach().mean().item()),
            "advantage_weight": float(
                advantage_weights.detach().mean().item()
            ),
            "actor_coefficient": float(
                coefficients.detach().mean().item()
            ),
        }

    def update_critic(self, batch: TorchBatch) -> Dict[str, float]:
        with torch.no_grad():
            next_values = self.value(batch["next_observations"])
            target_q = (
                batch["rewards"]
                + self.discount * batch["masks"] * next_values
            )

        q1, q2 = self.critic(
            batch["observations"],
            batch["actions"],
        )
        # The official IQL implementation sums the two squared errors before
        # taking the batch mean; it does not average over the two critics.
        critic_loss = (
            (q1 - target_q).pow(2) + (q2 - target_q).pow(2)
        ).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "q1": float(q1.detach().mean().item()),
            "q2": float(q2.detach().mean().item()),
        }

    def update(
        self,
        batch: TorchBatch,
        random_coefficients: bool,
    ) -> Dict[str, float]:
        """Run one official FamO2O-IQL gradient iteration.

        Update order:
          1. IQL value network.
          2. Balance policy, when due.
          3. Balance entropy temperature, when the balance policy is updated.
          4. Universal coefficient-conditioned policy.
          5. Double Q critic.
          6. Target-Q Polyak update.
        """

        self.update_step += 1

        # Draw coefficients before optimizing the balance policy so the online
        # actor update observes the old balance model, matching JAX semantics.
        universal_coefficients = self._sample_universal_coefficients(
            batch["observations"],
            random_coefficients=random_coefficients,
        )

        value_info = self.update_value(batch)

        balance_info: Dict[str, float] = {}
        temperature_info: Dict[str, float] = {}
        should_update_balance = (
            (self.update_step - 1) % self.balance_update_every == 0
        )
        if should_update_balance:
            balance_info, entropy = self.update_balance_policy(batch)
            temperature_info = self.update_temperature(entropy)

        actor_info = self.update_universal_policy(
            batch,
            coefficients=universal_coefficients,
        )
        critic_info = self.update_critic(batch)
        _soft_update(self.target_critic, self.critic, self.tau)

        return {
            **balance_info,
            **temperature_info,
            **actor_info,
            **critic_info,
            **value_info,
        }


# -----------------------------------------------------------------------------
# Batch conversion and evaluation
# -----------------------------------------------------------------------------


def _to_torch_batch(
    batch: Mapping[str, np.ndarray],
    device: torch.device,
) -> TorchBatch:
    return {
        "observations": torch.as_tensor(
            batch["observations"],
            dtype=torch.float32,
            device=device,
        ),
        "actions": torch.as_tensor(
            batch["actions"],
            dtype=torch.float32,
            device=device,
        ),
        "rewards": torch.as_tensor(
            batch["rewards"],
            dtype=torch.float32,
            device=device,
        ).reshape(-1),
        "masks": torch.as_tensor(
            batch["masks"],
            dtype=torch.float32,
            device=device,
        ).reshape(-1),
        "next_observations": torch.as_tensor(
            batch["next_observations"],
            dtype=torch.float32,
            device=device,
        ),
    }


@torch.no_grad()
def evaluate(
    learner: FamO2OLearner,
    env: gym.Env,
    num_episodes: int,
) -> Dict[str, float]:
    returns = []
    lengths = []

    learner.universal_policy.eval()
    learner.balance_policy.eval()
    for _ in range(num_episodes):
        observation = _reset_env(env)
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            # The official evaluator calls sample_actions with temperature=0,
            # which makes both policy levels deterministic.
            action = learner.sample_action(observation, temperature=0.0)
            observation, reward, done, _ = _step_env(env, action)
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)

    learner.universal_policy.train()
    learner.balance_policy.train()
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
    }


def _get_normalized_score(
    env: gym.Env,
    raw_return: float,
) -> Optional[float]:
    candidate = env
    for _ in range(16):
        if hasattr(candidate, "get_normalized_score"):
            try:
                return float(
                    candidate.get_normalized_score(raw_return) * 100.0
                )
            except Exception:
                return None
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    return None


# -----------------------------------------------------------------------------
# Configuration, seeding, and training
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


def _resolve_domain_defaults(config: TrainConfig) -> None:
    if _is_antmaze_env(config.env_name):
        default_expectile = 0.9
        default_minimum = 8.0
        default_maximum = 14.0
        default_balance_frequency = 1
    elif _is_locomotion_env(config.env_name):
        default_expectile = 0.7
        default_minimum = 1.0
        default_maximum = 5.0
        default_balance_frequency = 5
    else:
        # The official IQL+FamO2O paper/code publishes settings for D4RL
        # Locomotion and AntMaze. Other domains require explicit values rather
        # than silently pretending to use an official preset.
        missing = []
        if config.expectile is None:
            missing.append("expectile")
        if config.family_coefficient_min is None:
            missing.append("family_coefficient_min")
        if config.family_coefficient_max is None:
            missing.append("family_coefficient_max")
        if config.balance_update_every is None:
            missing.append("balance_update_every")
        if missing:
            raise ValueError(
                "No published IQL+FamO2O preset is available for "
                f"{config.env_name}. Set these fields explicitly: "
                + ", ".join(missing)
            )
        return

    if config.expectile is None:
        config.expectile = default_expectile
    if config.family_coefficient_min is None:
        config.family_coefficient_min = default_minimum
    if config.family_coefficient_max is None:
        config.family_coefficient_max = default_maximum
    if config.balance_update_every is None:
        config.balance_update_every = default_balance_frequency


def _validate_config(config: TrainConfig) -> None:
    _resolve_domain_defaults(config)

    if config.hidden_dim < 1 or config.hidden_layers < 1:
        raise ValueError("hidden_dim and hidden_layers must be positive.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if config.replay_buffer_size < 1:
        raise ValueError("replay_buffer_size must be positive.")
    if config.pretrain_steps < 0 or config.num_total_steps < 0:
        raise ValueError("Training-step counts cannot be negative.")
    if not 0.0 < config.discount <= 1.0:
        raise ValueError("discount must be in (0, 1].")
    if not 0.0 < config.tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")
    if config.expectile is None or not 0.0 < config.expectile < 1.0:
        raise ValueError("expectile must be in (0, 1).")
    if (
        config.family_coefficient_min is None
        or config.family_coefficient_max is None
        or config.family_coefficient_min >= config.family_coefficient_max
    ):
        raise ValueError(
            "family_coefficient_min must be smaller than "
            "family_coefficient_max."
        )
    if config.balance_update_every is None or config.balance_update_every < 1:
        raise ValueError("balance_update_every must be positive.")
    if config.family_sin_cos_d < 2 or config.family_sin_cos_d % 2 != 0:
        raise ValueError("family_sin_cos_d must be a positive even integer.")
    if config.init_temperature <= 0.0:
        raise ValueError("init_temperature must be positive.")
    if config.max_advantage_weight <= 0.0:
        raise ValueError("max_advantage_weight must be positive.")


def _should_evaluate(
    training_index: int,
    first_training_index: int,
    config: TrainConfig,
) -> bool:
    near_end_of_online = (
        training_index
        >= config.num_total_steps - config.last_eval_start_steps
    )
    near_end_of_offline = (
        -config.last_eval_start_steps <= training_index < 1
    )
    interval = (
        config.last_eval_every
        if near_end_of_online or near_end_of_offline
        else config.eval_every
    )
    return training_index == first_training_index or training_index % interval == 0


def train(config: TrainConfig) -> None:
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)
    device = torch.device(config.device)

    train_env = wrap_gym(gym.make(config.env_name))
    eval_env = wrap_gym(gym.make(config.env_name))
    _seed_env(train_env, config.train_seed)
    _seed_env(eval_env, config.train_seed + config.eval_seed_offset)

    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise TypeError("The wrapped observation space must be Box.")
    if not isinstance(train_env.action_space, gym.spaces.Box):
        raise TypeError("The wrapped action space must be Box.")

    offline_dataset = D4RLDataset(train_env)
    if config.replay_buffer_size < offline_dataset.size:
        raise ValueError(
            f"Replay capacity {config.replay_buffer_size} is smaller than "
            f"the offline dataset ({offline_dataset.size})."
        )

    locomotion_return_range = None
    if _is_locomotion_env(config.env_name):
        locomotion_return_range = _compute_locomotion_return_range(
            offline_dataset
        )

    replay_buffer = ReplayBuffer(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        capacity=config.replay_buffer_size,
        seed=config.train_seed,
    )
    replay_buffer.initialize_with_dataset(
        offline_dataset,
        num_samples=config.init_dataset_size,
    )

    observation_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))
    hidden_dims = tuple(
        config.hidden_dim for _ in range(config.hidden_layers)
    )

    universal_policy = UniversalPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        encoding_dim=config.family_sin_cos_d,
        hidden_dims=hidden_dims,
        log_std_min=config.universal_log_std_min,
        log_std_max=config.universal_log_std_max,
    ).to(device)
    balance_policy = BalancePolicy(
        observation_dim=observation_dim,
        hidden_dims=hidden_dims,
        log_std_min=config.balance_log_std_min,
        log_std_max=config.balance_log_std_max,
    ).to(device)
    critic = DoubleCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    value = ValueNetwork(
        observation_dim=observation_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    universal_optimizer = torch.optim.Adam(
        universal_policy.parameters(),
        lr=config.actor_learning_rate,
    )
    balance_optimizer = torch.optim.Adam(
        balance_policy.parameters(),
        lr=config.balance_learning_rate,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.critic_learning_rate,
    )
    value_optimizer = torch.optim.Adam(
        value.parameters(),
        lr=config.value_learning_rate,
    )

    target_entropy = (
        -0.5
        if config.target_entropy is None
        else float(config.target_entropy)
    )
    learner = FamO2OLearner(
        universal_policy=universal_policy,
        balance_policy=balance_policy,
        critic=critic,
        value=value,
        universal_optimizer=universal_optimizer,
        balance_optimizer=balance_optimizer,
        critic_optimizer=critic_optimizer,
        value_optimizer=value_optimizer,
        temperature_learning_rate=config.temperature_learning_rate,
        init_temperature=config.init_temperature,
        target_entropy=target_entropy,
        discount=config.discount,
        tau=config.tau,
        expectile=float(config.expectile),
        coefficient_range=(
            float(config.family_coefficient_min),
            float(config.family_coefficient_max),
        ),
        family_sin_cos_n=config.family_sin_cos_n,
        family_sin_cos_d=config.family_sin_cos_d,
        max_advantage_weight=config.max_advantage_weight,
        balance_update_every=int(config.balance_update_every),
        device=device,
    )

    run_name = f"{config.env_name}_seed{config.train_seed}"
    log_dir = os.path.join(
        config.log_root,
        config.env_name.split("-")[0],
        run_name,
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"env={config.env_name}\n"
        f"device={device}\n"
        f"base_algorithm=IQL\n"
        f"hidden_dims={hidden_dims}\n"
        f"expectile={config.expectile}\n"
        f"coefficient_range=[{config.family_coefficient_min}, "
        f"{config.family_coefficient_max}]\n"
        f"balance_update_every={config.balance_update_every}\n"
        f"batch_size={config.batch_size}, replay={config.replay_buffer_size}\n"
        f"offline_steps={config.pretrain_steps}, "
        f"online_steps={config.num_total_steps}\n"
        f"initial_replay_size={replay_buffer.size}\n"
        f"log_dir={log_dir}",
        flush=True,
    )

    observation = _reset_env(train_env)
    episode_return = 0.0
    episode_length = 0

    first_training_index = 1 - config.pretrain_steps
    last_training_index = config.num_total_steps

    for training_index in trange(
        first_training_index,
        last_training_index + 1,
        desc="FamO2O IQL pretraining/fine-tuning",
    ):
        is_online = training_index >= 1

        if is_online:
            action = learner.sample_action(observation, temperature=1.0)
            next_observation, reward, done, info = _step_env(
                train_env,
                action,
            )
            timeout = "TimeLimit.truncated" in info
            mask = 1.0 if (not done or timeout) else 0.0

            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                mask=mask,
                done_float=float(done),
                next_observation=next_observation,
            )

            episode_return += reward
            episode_length += 1
            observation = next_observation

            if done:
                writer.add_scalar(
                    "online-training/return",
                    episode_return,
                    training_index,
                )
                writer.add_scalar(
                    "online-training/length",
                    episode_length,
                    training_index,
                )
                observation = _reset_env(train_env)
                episode_return = 0.0
                episode_length = 0

        numpy_batch = replay_buffer.sample(config.batch_size)
        numpy_batch = _transform_batch_rewards(
            numpy_batch,
            config.env_name,
            locomotion_return_range,
        )
        torch_batch = _to_torch_batch(numpy_batch, device)
        metrics = learner.update(
            torch_batch,
            random_coefficients=not is_online,
        )

        if training_index % config.log_every == 0:
            phase = "online-training" if is_online else "offline-training"
            for key, value_metric in metrics.items():
                writer.add_scalar(
                    f"{phase}/{key}",
                    value_metric,
                    training_index,
                )
            writer.add_scalar(
                f"{phase}/replay_size",
                replay_buffer.size,
                training_index,
            )

        if _should_evaluate(
            training_index,
            first_training_index,
            config,
        ):
            eval_info = evaluate(
                learner,
                eval_env,
                config.eval_episodes,
            )
            writer.add_scalar(
                "evaluation/return",
                eval_info["return"],
                training_index,
            )
            writer.add_scalar(
                "evaluation/length",
                eval_info["length"],
                training_index,
            )
            normalized_score = _get_normalized_score(
                eval_env,
                eval_info["return"],
            )
            message = (
                f"[eval] step={training_index}, "
                f"phase={'online' if is_online else 'offline'}, "
                f"return={eval_info['return']:.3f}, "
                f"length={eval_info['length']:.1f}"
            )
            if normalized_score is not None:
                writer.add_scalar(
                    "evaluation/d4rl_normalized_score",
                    normalized_score,
                    training_index,
                )
                message += f", normalized_score={normalized_score:.3f}"
            print(message, flush=True)
            writer.flush()

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "universal_policy": universal_policy.state_dict(),
                "balance_policy": balance_policy.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": learner.target_critic.state_dict(),
                "value": value.state_dict(),
                "log_temperature": learner.log_temperature.detach().cpu(),
                "universal_optimizer": universal_optimizer.state_dict(),
                "balance_optimizer": balance_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "value_optimizer": value_optimizer.state_dict(),
                "temperature_optimizer": (
                    learner.temperature_optimizer.state_dict()
                ),
                "config": asdict(config),
            },
            os.path.join(checkpoint_dir, "final.pt"),
        )

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    config = pyrallis.parse(config_class=TrainConfig)
    train(config)
