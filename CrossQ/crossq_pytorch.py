"""Single-file PyTorch reproduction of CrossQ.

This implementation follows the ICLR 2024 CrossQ paper and the authors'
official JAX/Flax repository.  It keeps the same compact single-file style as
the user's other PyTorch reproductions while preserving the implementation
choices that distinguish CrossQ from ordinary SAC:

* no target critic network;
* two independent critics;
* Batch Renormalization (BRN) in both actor and critics;
* one joint critic forward pass over current and next state-action pairs;
* UTD ratio 1 and policy delay 3;
* 2048-unit critic layers and 256-unit actor layers;
* Adam beta_1 = 0.5 for actor and critic;
* automatic entropy-temperature optimization;
* BRN moving-average momentum 0.99 and a 100,000-update BRN warm-up.

The official implementation is written in JAX/Flax.  PyTorch and JAX use
different PRNGs, parameter layouts, and low-level kernels, so this file targets
algorithmic and implementation-level equivalence rather than bitwise equality.
"""

from __future__ import annotations

import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import trange

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility with older installations
    try:
        import gym  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - network-only tests can still import the file
        gym = None  # type: ignore[assignment]

try:
    import pyrallis
except ImportError:  # pragma: no cover
    pyrallis = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


NumpyBatch = Dict[str, np.ndarray]
TorchBatch = Dict[str, torch.Tensor]


@dataclass
class TrainConfig:
    # The official README demonstrates CrossQ on Humanoid-v4.
    env_name: str = "Humanoid-v4"

    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    critic_hidden_dims: Tuple[int, ...] = (2048, 2048)
    num_critics: int = 2

    discount: Optional[float] = None  # official: 0.99, except Swimmer-v4=0.9999
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    temperature_learning_rate: float = 1e-3
    actor_adam_beta1: float = 0.5
    critic_adam_beta1: float = 0.5
    temperature_adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    init_temperature: float = 1.0
    target_entropy: Optional[float] = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # Flax calls this the moving-average momentum:
    # running <- momentum * running + (1 - momentum) * batch.
    brn_momentum: float = 0.99
    brn_epsilon: float = 1e-3
    brn_warmup_steps: int = 100_000
    brn_r_max: float = 3.0
    brn_d_max: float = 5.0

    replay_buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    utd_ratio: int = 1
    policy_delay: int = 3
    num_total_steps: Optional[int] = None  # official MuJoCo default: 5,000,000

    eval_episodes: int = 1
    eval_every: Optional[int] = None  # official default: approximately 300 evaluations
    log_every: int = 1_000

    train_seed: int = 1
    eval_seed_offset: int = 10_000
    deterministic_torch: bool = False

    checkpoints_path: Optional[str] = None
    log_root: str = "logs/CrossQ_PyTorch"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Utilities and environment compatibility
# -----------------------------------------------------------------------------


def _require_training_dependencies() -> None:
    missing = []
    if gym is None:
        missing.append("gymnasium (or gym)")
    if pyrallis is None:
        missing.append("pyrallis")
    if SummaryWriter is None:
        missing.append("tensorboard")
    if missing:
        raise ImportError(
            "Missing training dependencies: " + ", ".join(missing) + "."
        )


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


def _resolve_total_steps(config: TrainConfig) -> int:
    if config.num_total_steps is not None:
        return int(config.num_total_steps)
    dm_control_steps = {
        "dm_control/reacher-easy": 100_000,
        "dm_control/reacher-hard": 100_000,
        "dm_control/ball_in_cup-catch": 200_000,
        "dm_control/finger-spin": 500_000,
        "dm_control/fish-swim": 5_000_000,
        "dm_control/humanoid-stand": 5_000_000,
    }
    return dm_control_steps.get(config.env_name, 5_000_000)


def _resolve_discount(config: TrainConfig) -> float:
    if config.discount is not None:
        return float(config.discount)
    return 0.9999 if config.env_name == "Swimmer-v4" else 0.99


def _resolve_eval_every(config: TrainConfig, total_steps: int) -> int:
    if config.eval_every is not None:
        return max(int(config.eval_every), 1)
    return max(total_steps // 300, 1)


def _validate_config(config: TrainConfig) -> None:
    if config.num_critics != 2:
        raise ValueError("The official CrossQ configuration uses exactly two critics.")
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for Batch Renormalization.")
    if config.replay_buffer_size < config.batch_size:
        raise ValueError("replay_buffer_size must be at least batch_size.")
    if config.learning_starts < config.batch_size:
        raise ValueError("learning_starts must be at least batch_size.")
    if config.utd_ratio < 1 or config.policy_delay < 1:
        raise ValueError("utd_ratio and policy_delay must be positive.")
    if not 0.0 <= config.brn_momentum < 1.0:
        raise ValueError("brn_momentum must lie in [0, 1).")
    if config.brn_warmup_steps < 0:
        raise ValueError("brn_warmup_steps cannot be negative.")


def _make_env(env_name: str):
    if gym is None:
        raise ImportError("Install gymnasium or gym before starting training.")
    env = gym.make(env_name)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("CrossQ requires a flat continuous Box observation space.")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("CrossQ requires a continuous Box action space.")

    # SB3/SBX stores and predicts normalized actions in [-1, 1].
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)
    if not (np.allclose(low, -1.0) and np.allclose(high, 1.0)):
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    env = gym.wrappers.ClipAction(env)
    return env


def _seed_env(env, seed: int) -> None:
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass


def _reset_env(env, seed: Optional[int] = None) -> np.ndarray:
    try:
        output = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:  # old Gym
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        output = env.reset()
    observation = output[0] if isinstance(output, tuple) else output
    return np.asarray(observation, dtype=np.float32)


def _step_env(env, action: np.ndarray):
    output = env.step(action)
    if len(output) == 5:
        next_observation, reward, terminated, truncated, info = output
        done = bool(terminated or truncated)
        terminal = bool(terminated)
        return (
            np.asarray(next_observation, dtype=np.float32),
            float(reward),
            done,
            terminal,
            dict(info),
        )

    next_observation, reward, done, info = output
    info = dict(info)
    timeout = bool(info.get("TimeLimit.truncated", False))
    terminal = bool(done and not timeout)
    return (
        np.asarray(next_observation, dtype=np.float32),
        float(reward),
        bool(done),
        terminal,
        info,
    )


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_shape: Sequence[int],
        capacity: int,
        seed: int,
    ):
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.observations = np.empty(
            (capacity, *observation_shape), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.terminals = np.empty((capacity,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        next_observation: np.ndarray,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = float(terminal)
        self.next_observations[index] = next_observation
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size < batch_size:
            raise ValueError(
                f"Replay contains {self.size} transitions, fewer than {batch_size}."
            )
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "terminals": self.terminals[indices],
            "next_observations": self.next_observations[indices],
        }


def _to_torch_batch(batch: NumpyBatch, device: torch.device) -> TorchBatch:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in batch.items()
    }


# -----------------------------------------------------------------------------
# Flax-equivalent initialization and Batch Renormalization
# -----------------------------------------------------------------------------


def _lecun_normal_(layer: nn.Linear) -> None:
    fan_in = layer.weight.shape[1]
    nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BatchRenorm1d(nn.Module):
    """PyTorch translation of the BRN module in the official CrossQ code.

    ``moving_average_momentum`` follows the Flax convention:
    running <- momentum * running + (1 - momentum) * batch.
    """

    def __init__(
        self,
        num_features: int,
        moving_average_momentum: float = 0.99,
        epsilon: float = 1e-3,
        warmup_steps: int = 100_000,
        r_max: float = 3.0,
        d_max: float = 5.0,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.moving_average_momentum = float(moving_average_momentum)
        self.epsilon = float(epsilon)
        self.warmup_steps = int(warmup_steps)
        self.r_max = float(r_max)
        self.d_max = float(d_max)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked", torch.zeros((), dtype=torch.long)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        use_running_average: bool,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(
                f"BatchRenorm1d expects [batch, features], got {tuple(inputs.shape)}."
            )

        if use_running_average:
            mean = self.running_mean
            variance = self.running_var
        else:
            mean = inputs.mean(dim=0)
            variance = inputs.var(dim=0, unbiased=False)
            custom_mean = mean
            custom_variance = variance

            if int(self.num_batches_tracked.item()) >= self.warmup_steps:
                running_std = torch.sqrt(self.running_var + self.epsilon)
                batch_std = torch.sqrt(variance + self.epsilon)
                r = (batch_std / running_std).detach()
                r = torch.clamp(r, 1.0 / self.r_max, self.r_max)
                d = ((mean - self.running_mean) / running_std).detach()
                d = torch.clamp(d, -self.d_max, self.d_max)

                # Match the custom moments used by the official Flax module.
                custom_variance = variance / r.pow(2)
                custom_mean = mean - d * torch.sqrt(variance) / r

            mean = custom_mean
            variance = custom_variance

            with torch.no_grad():
                momentum = self.moving_average_momentum
                self.running_mean.mul_(momentum).add_(
                    inputs.mean(dim=0), alpha=1.0 - momentum
                )
                self.running_var.mul_(momentum).add_(
                    inputs.var(dim=0, unbiased=False), alpha=1.0 - momentum
                )
                self.num_batches_tracked.add_(1)

        normalized = (inputs - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


# -----------------------------------------------------------------------------
# Actor and critics
# -----------------------------------------------------------------------------


def _tanh_log_prob(distribution: Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
    base_log_prob = distribution.log_prob(pre_tanh)
    correction = 2.0 * (
        math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
    )
    return (base_log_prob - correction).sum(dim=-1)


class CrossQActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        brn_momentum: float,
        brn_epsilon: float,
        brn_warmup_steps: int,
        brn_r_max: float,
        brn_d_max: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.input_norm = BatchRenorm1d(
            observation_dim,
            brn_momentum,
            brn_epsilon,
            brn_warmup_steps,
            brn_r_max,
            brn_d_max,
        )
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()

        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(input_dim, hidden_dim)
            _lecun_normal_(layer)
            self.hidden_layers.append(layer)
            self.hidden_norms.append(
                BatchRenorm1d(
                    hidden_dim,
                    brn_momentum,
                    brn_epsilon,
                    brn_warmup_steps,
                    brn_r_max,
                    brn_d_max,
                )
            )
            input_dim = hidden_dim

        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)
        _lecun_normal_(self.mean)
        _lecun_normal_(self.log_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def distribution(
        self,
        observations: torch.Tensor,
        train: bool,
    ) -> Normal:
        values = self.input_norm(observations, use_running_average=not train)
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            values = F.relu(layer(values))
            values = norm(values, use_running_average=not train)
        mean = self.mean(values)
        log_std = self.log_std(values).clamp(
            min=self.log_std_min, max=self.log_std_max
        )
        return Normal(mean, log_std.exp())

    def sample(
        self,
        observations: torch.Tensor,
        train: bool,
        reparameterize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations, train=train)
        pre_tanh = (
            distribution.rsample() if reparameterize else distribution.sample()
        )
        actions = torch.tanh(pre_tanh)
        log_probability = _tanh_log_prob(distribution, pre_tanh)
        return actions, log_probability

    def mode(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        distribution = self.distribution(observations, train=train)
        return torch.tanh(distribution.mean)

    @torch.no_grad()
    def sample_action(
        self,
        observation: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action, _ = self.sample(tensor, train=False, reparameterize=False)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def eval_action(
        self,
        observation: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        return self.mode(tensor, train=False).squeeze(0).cpu().numpy()


class CrossQCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        brn_momentum: float,
        brn_epsilon: float,
        brn_warmup_steps: int,
        brn_r_max: float,
        brn_d_max: float,
    ):
        super().__init__()
        input_dim = observation_dim + action_dim
        self.input_norm = BatchRenorm1d(
            input_dim,
            brn_momentum,
            brn_epsilon,
            brn_warmup_steps,
            brn_r_max,
            brn_d_max,
        )
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()

        for hidden_dim in hidden_dims:
            layer = nn.Linear(input_dim, hidden_dim)
            _lecun_normal_(layer)
            self.hidden_layers.append(layer)
            self.hidden_norms.append(
                BatchRenorm1d(
                    hidden_dim,
                    brn_momentum,
                    brn_epsilon,
                    brn_warmup_steps,
                    brn_r_max,
                    brn_d_max,
                )
            )
            input_dim = hidden_dim

        self.output_layer = nn.Linear(input_dim, 1)
        _lecun_normal_(self.output_layer)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        values = torch.cat([observations, actions], dim=-1)
        values = self.input_norm(values, use_running_average=not train)
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            values = F.relu(layer(values))
            values = norm(values, use_running_average=not train)
        return self.output_layer(values).squeeze(-1)


class DoubleCrossQCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        num_critics: int,
        brn_momentum: float,
        brn_epsilon: float,
        brn_warmup_steps: int,
        brn_r_max: float,
        brn_d_max: float,
    ):
        super().__init__()
        self.critics = nn.ModuleList(
            [
                CrossQCritic(
                    observation_dim,
                    action_dim,
                    hidden_dims,
                    brn_momentum,
                    brn_epsilon,
                    brn_warmup_steps,
                    brn_r_max,
                    brn_d_max,
                )
                for _ in range(num_critics)
            ]
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        return torch.stack(
            [critic(observations, actions, train=train) for critic in self.critics],
            dim=0,
        )


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


# -----------------------------------------------------------------------------
# CrossQ learner
# -----------------------------------------------------------------------------


class CrossQLearner:
    def __init__(
        self,
        actor: CrossQActor,
        critic: DoubleCrossQCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temperature_learning_rate: float,
        temperature_adam_beta1: float,
        adam_beta2: float,
        init_temperature: float,
        target_entropy: float,
        discount: float,
        policy_delay: int,
        device: torch.device,
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.log_temperature = torch.tensor(
            math.log(init_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature],
            lr=temperature_learning_rate,
            betas=(temperature_adam_beta1, adam_beta2),
        )
        self.target_entropy = float(target_entropy)
        self.discount = float(discount)
        self.policy_delay = int(policy_delay)
        self.device = device
        self.num_critic_updates = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def update_critic(self, batch: TorchBatch) -> Dict[str, float]:
        batch_size = batch["observations"].shape[0]

        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(
                batch["next_observations"],
                train=False,
                reparameterize=False,
            )

        # CrossQ's defining joint forward pass.  Current and next pairs share
        # exactly the same BRN minibatch statistics.
        all_observations = torch.cat(
            [batch["observations"], batch["next_observations"]], dim=0
        )
        all_actions = torch.cat([batch["actions"], next_actions], dim=0)
        all_q_values = self.critic(all_observations, all_actions, train=True)
        current_q_values = all_q_values[:, :batch_size]
        next_q_values = all_q_values[:, batch_size:]

        next_q = next_q_values.min(dim=0).values
        next_q = next_q - self.temperature.detach() * next_log_probability
        target_q = (
            batch["rewards"]
            + (1.0 - batch["terminals"]) * self.discount * next_q
        ).detach()

        # Official JAX code: 0.5 * mean_per_critic(...).sum().
        critic_loss = 0.5 * (
            (current_q_values - target_q.unsqueeze(0)).pow(2).mean(dim=1)
        ).sum()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.num_critic_updates += 1

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "current_q": float(current_q_values.detach().mean().item()),
            "next_q": float(next_q_values.detach().mean().item()),
        }

    def update_actor_and_temperature(self, batch: TorchBatch) -> Dict[str, float]:
        actions, log_probability = self.actor.sample(
            batch["observations"], train=True, reparameterize=True
        )
        with _frozen_parameters(self.critic):
            q_values = self.critic(
                batch["observations"], actions, train=False
            ).min(dim=0).values

        actor_loss = (
            self.temperature.detach() * log_probability - q_values
        ).mean()
        entropy = -log_probability.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        temperature_before = self.temperature
        temperature_loss = temperature_before * (
            entropy.detach() - self.target_entropy
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "temperature": float(temperature_before.detach().item()),
            "temperature_loss": float(temperature_loss.detach().item()),
        }

    def update(self, super_batch: TorchBatch, utd_ratio: int) -> Dict[str, float]:
        if utd_ratio < 1:
            raise ValueError("utd_ratio must be positive.")
        total_size = next(iter(super_batch.values())).shape[0]
        if total_size % utd_ratio != 0:
            raise ValueError("Super-batch size must be divisible by utd_ratio.")

        mini_batch_size = total_size // utd_ratio
        metrics: Dict[str, float] = {}
        for update_index in range(utd_ratio):
            start = update_index * mini_batch_size
            end = start + mini_batch_size
            mini_batch = {
                key: value[start:end] for key, value in super_batch.items()
            }
            metrics.update(self.update_critic(mini_batch))
            if self.num_critic_updates % self.policy_delay == 0:
                metrics.update(self.update_actor_and_temperature(mini_batch))
        metrics["critic_updates"] = float(self.num_critic_updates)
        return metrics


# -----------------------------------------------------------------------------
# Evaluation and training
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    actor: CrossQActor,
    env,
    device: torch.device,
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
            action = actor.eval_action(observation, device)
            observation, reward, done, _, _ = _step_env(env, action)
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
    }


def train(config: TrainConfig) -> None:
    _require_training_dependencies()
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)

    device = torch.device(config.device)
    total_steps = _resolve_total_steps(config)
    discount = _resolve_discount(config)
    eval_every = _resolve_eval_every(config, total_steps)

    train_env = _make_env(config.env_name)
    eval_env = _make_env(config.env_name)
    _seed_env(train_env, config.train_seed)
    _seed_env(eval_env, config.train_seed + config.eval_seed_offset)

    observation_shape = tuple(train_env.observation_space.shape)
    action_shape = tuple(train_env.action_space.shape)
    observation_dim = int(np.prod(observation_shape))
    action_dim = int(np.prod(action_shape))

    actor = CrossQActor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=config.actor_hidden_dims,
        brn_momentum=config.brn_momentum,
        brn_epsilon=config.brn_epsilon,
        brn_warmup_steps=config.brn_warmup_steps,
        brn_r_max=config.brn_r_max,
        brn_d_max=config.brn_d_max,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
    ).to(device)
    critic = DoubleCrossQCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=config.critic_hidden_dims,
        num_critics=config.num_critics,
        brn_momentum=config.brn_momentum,
        brn_epsilon=config.brn_epsilon,
        brn_warmup_steps=config.brn_warmup_steps,
        brn_r_max=config.brn_r_max,
        brn_d_max=config.brn_d_max,
    ).to(device)

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=config.actor_learning_rate,
        betas=(config.actor_adam_beta1, config.adam_beta2),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.critic_learning_rate,
        betas=(config.critic_adam_beta1, config.adam_beta2),
    )
    target_entropy = (
        -float(action_dim)
        if config.target_entropy is None
        else float(config.target_entropy)
    )
    learner = CrossQLearner(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        temperature_learning_rate=config.temperature_learning_rate,
        temperature_adam_beta1=config.temperature_adam_beta1,
        adam_beta2=config.adam_beta2,
        init_temperature=config.init_temperature,
        target_entropy=target_entropy,
        discount=discount,
        policy_delay=config.policy_delay,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        observation_shape=observation_shape,
        action_shape=action_shape,
        capacity=config.replay_buffer_size,
        seed=config.train_seed,
    )

    run_name = f"{config.env_name}_seed{config.train_seed}"
    log_dir = os.path.join(config.log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(
        f"env={config.env_name}\n"
        f"device={device}\n"
        f"actor={config.actor_hidden_dims}, critic={config.critic_hidden_dims}\n"
        f"critics={config.num_critics}, UTD={config.utd_ratio}, "
        f"policy_delay={config.policy_delay}\n"
        f"BRN momentum={config.brn_momentum}, "
        f"warmup={config.brn_warmup_steps}\n"
        f"discount={discount}, total_steps={total_steps}\n"
        f"log_dir={log_dir}",
        flush=True,
    )

    observation = _reset_env(train_env, seed=config.train_seed)
    episode_return = 0.0
    episode_length = 0

    for step in trange(1, total_steps + 1, desc="CrossQ training"):
        if step <= config.learning_starts:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            action = actor.sample_action(observation, device)

        next_observation, reward, done, terminal, _ = _step_env(train_env, action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
        )

        episode_return += reward
        episode_length += 1
        observation = next_observation

        if done:
            writer.add_scalar("training/return", episode_return, step)
            writer.add_scalar("training/length", episode_length, step)
            observation = _reset_env(train_env)
            episode_return = 0.0
            episode_length = 0

        metrics: Dict[str, float] = {}
        if step > config.learning_starts:
            super_batch = replay_buffer.sample(
                config.batch_size * config.utd_ratio
            )
            metrics = learner.update(
                _to_torch_batch(super_batch, device), config.utd_ratio
            )

        if metrics and step % config.log_every == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"training/{key}", value, step)

        if step % eval_every == 0:
            eval_metrics = evaluate(
                actor, eval_env, device, config.eval_episodes
            )
            writer.add_scalar("evaluation/return", eval_metrics["return"], step)
            writer.add_scalar("evaluation/length", eval_metrics["length"], step)
            print(
                f"[eval] step={step}, return={eval_metrics['return']:.3f}, "
                f"length={eval_metrics['length']:.1f}",
                flush=True,
            )
            writer.flush()

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "log_temperature": learner.log_temperature.detach().cpu(),
                "num_critic_updates": learner.num_critic_updates,
                "config": asdict(config),
                "resolved_discount": discount,
                "resolved_total_steps": total_steps,
            },
            os.path.join(checkpoint_dir, "final.pt"),
        )

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    _require_training_dependencies()
    config = pyrallis.parse(config_class=TrainConfig)
    train(config)
