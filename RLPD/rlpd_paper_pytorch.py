"""Single-file PyTorch implementation of the state-based RLPD algorithm.

The implementation keeps the user's existing single-file PyTorch style while
following the two primary RLPD sources deliberately:

1. The released JAX/Flax repository is used for network definitions, SAC
   objectives, D4RL preprocessing, 50/50 offline-online interleaving, target
   updates, and experiment hyperparameters.
2. Algorithm 1 in the RLPD paper is used for the asymmetric update schedule:
   each environment step performs ``utd_ratio`` sequential critic updates,
   followed by one actor update and one temperature update on the final
   mini-batch.

The paper pseudocode and the released JAX learner differ on this point: the
repository updates critic, actor, and temperature inside every UTD iteration,
whereas Algorithm 1 places the actor update after the critic-update loop. This
file intentionally implements the paper schedule requested by the user.

The default configuration is the paper/README AntMaze setting: 10 critics,
UTD=20, three 256-unit hidden layers, ``num_min_qs=1``, no entropy backup,
5,000 random-action steps, and 300,000 online steps.

PyTorch and JAX/Flax use different PRNG implementations, parameter layouts, and
low-level kernels. Therefore, this is an algorithmically faithful PyTorch
implementation, not a bitwise-identical execution for the same numeric seed.
"""

import math
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Avoid irrelevant import errors for optional D4RL domains such as Flow/CARLA.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import d4rl  # noqa: F401: importing d4rl registers its Gym environments
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

try:
    from rlpd.data.binary_datasets import BinaryDataset
except Exception:
    BinaryDataset = None


NumpyBatch = Dict[str, np.ndarray]
TorchBatch = Dict[str, torch.Tensor]


@dataclass
class TrainConfig:
    # RLPD paper/README AntMaze reference configuration.
    env_name: str = "antmaze-umaze-v2"
    dataset_source: str = "auto"  # auto | d4rl | binary
    binary_include_bc: bool = True

    hidden_dim: int = 256
    hidden_layers: int = 3
    num_critics: int = 10
    num_min_qs: Optional[int] = 1
    critic_layer_norm: bool = True
    critic_weight_decay: Optional[float] = None

    discount: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None
    backup_entropy: bool = False

    batch_size: int = 256
    utd_ratio: int = 20
    offline_ratio: float = 0.5
    start_training: int = 5_000
    pretrain_steps: int = 0
    num_total_steps: int = 300_000

    eval_episodes: int = 10
    eval_every: int = 5_000
    log_every: int = 1_000

    train_seed: int = 42
    eval_seed_offset: int = 42
    offline_dataset_seed: Optional[int] = None
    deterministic_torch: bool = False

    checkpoints_path: Optional[str] = None
    log_root: str = "logs/RLPD_Paper_PyTorch"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Environment compatibility and official wrappers
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
    return next_observation, float(reward), bool(done), info


# -----------------------------------------------------------------------------
# Dataset and replay buffer
# -----------------------------------------------------------------------------


class ArrayDataset:
    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        seed: Optional[int] = None,
    ):
        if not arrays:
            raise ValueError("Dataset cannot be empty.")
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("Dataset fields have inconsistent lengths.")
        self.arrays = {key: np.asarray(value) for key, value in arrays.items()}
        self.size = lengths.pop()
        self._np_random, self.seed_value = gym.utils.seeding.np_random(seed)

    def sample(self, batch_size: int) -> NumpyBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if hasattr(self._np_random, "integers"):
            indices = self._np_random.integers(self.size, size=batch_size)
        else:
            indices = self._np_random.randint(self.size, size=batch_size)
        return {key: value[indices] for key, value in self.arrays.items()}


class D4RLDataset(ArrayDataset):
    def __init__(
        self,
        env: gym.Env,
        seed: Optional[int] = None,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        data = d4rl.qlearning_dataset(env)
        data = {key: np.asarray(value).copy() for key, value in data.items()}

        if clip_to_eps:
            limit = 1.0 - eps
            data["actions"] = np.clip(data["actions"], -limit, limit)

        rewards = np.asarray(data["rewards"])
        terminals = np.asarray(data["terminals"])
        dones = np.full_like(rewards, False, dtype=bool)
        for index in range(len(dones) - 1):
            discontinuity = (
                np.linalg.norm(
                    data["observations"][index + 1]
                    - data["next_observations"][index]
                )
                > 1e-6
            )
            if discontinuity or terminals[index] == 1.0:
                dones[index] = True
        dones[-1] = True

        data["masks"] = 1.0 - terminals
        del data["terminals"]

        for key, value in list(data.items()):
            data[key] = value.astype(np.float32)
        data["dones"] = dones

        super().__init__(data, seed=seed)


class ReplayBuffer:
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
            raise ValueError("Replay capacity must be positive.")

        self.capacity = int(capacity)
        self.size = 0
        self.insert_index = 0
        self.observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(
            (capacity, *action_space.shape), dtype=action_space.dtype
        )
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=bool)
        self._np_random, self.seed_value = gym.utils.seeding.np_random(seed)

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        index = self.insert_index
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.masks[index] = mask
        self.dones[index] = done
        self.next_observations[index] = next_observation
        self.insert_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if hasattr(self._np_random, "integers"):
            indices = self._np_random.integers(self.size, size=batch_size)
        else:
            indices = self._np_random.randint(self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "dones": self.dones[indices],
            "next_observations": self.next_observations[indices],
        }


class ExternalDatasetAdapter:
    def __init__(self, dataset):
        self.dataset = dataset

    def sample(self, batch_size: int) -> NumpyBatch:
        batch = self.dataset.sample(batch_size)
        return {key: np.asarray(value) for key, value in dict(batch).items()}


# -----------------------------------------------------------------------------
# Networks: Flax-equivalent Xavier initialization and LayerNorm epsilon
# -----------------------------------------------------------------------------


def _init_linear_xavier(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        layers = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(input_dim, hidden_dim)
            _init_linear_xavier(linear)
            layers.extend([linear, nn.ReLU()])
            input_dim = hidden_dim
        self.base = nn.Sequential(*layers)
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)
        _init_linear_xavier(self.mean)
        _init_linear_xavier(self.log_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def distribution(self, observations: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.base(observations)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            min=self.log_std_min, max=self.log_std_max
        )
        return Normal(mean, log_std.exp()), mean

    @staticmethod
    def _log_prob_from_pre_tanh(
        distribution: Normal,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        base_log_prob = distribution.log_prob(pre_tanh)
        log_abs_det_jacobian = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (base_log_prob - log_abs_det_jacobian).sum(dim=-1)

    def sample(
        self,
        observations: torch.Tensor,
        reparameterize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution, _ = self.distribution(observations)
        pre_tanh = (
            distribution.rsample() if reparameterize else distribution.sample()
        )
        actions = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(distribution, pre_tanh)
        return actions, log_prob

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        _, mean = self.distribution(observations)
        return torch.tanh(mean)

    @torch.no_grad()
    def sample_action(self, observation: np.ndarray, device: torch.device) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action, _ = self.sample(tensor, reparameterize=False)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def eval_action(self, observation: np.ndarray, device: torch.device) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action = self.mode(tensor)
        return action.squeeze(0).cpu().numpy()


class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(ensemble_size, input_dim, output_dim)
        )
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, output_dim))
        for ensemble_index in range(ensemble_size):
            nn.init.xavier_uniform_(self.weight[ensemble_index])

    def forward(
        self,
        inputs: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = self.weight if indices is None else self.weight.index_select(0, indices)
        bias = self.bias if indices is None else self.bias.index_select(0, indices)
        if inputs.ndim == 2:
            return torch.einsum("bi,eio->ebo", inputs, weight) + bias
        if inputs.ndim == 3:
            return torch.einsum("ebi,eio->ebo", inputs, weight) + bias
        raise ValueError(f"Expected 2D or 3D input, got shape {tuple(inputs.shape)}")


class EnsembleLayerNorm(nn.Module):
    def __init__(self, ensemble_size: int, feature_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, 1, feature_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, feature_dim))
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = self.weight if indices is None else self.weight.index_select(0, indices)
        bias = self.bias if indices is None else self.bias.index_select(0, indices)
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        return normalized * weight + bias


class EnsembleCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        ensemble_size: int,
        layer_norm: bool,
    ):
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        input_dim = observation_dim + action_dim
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(
                EnsembleLinear(self.ensemble_size, input_dim, hidden_dim)
            )
            if layer_norm:
                self.layer_norms.append(
                    EnsembleLayerNorm(self.ensemble_size, hidden_dim, eps=1e-6)
                )
            input_dim = hidden_dim
        self.use_layer_norm = bool(layer_norm)
        self.output_layer = EnsembleLinear(self.ensemble_size, input_dim, 1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        values = torch.cat([observations, actions], dim=-1)
        for layer_index, linear in enumerate(self.hidden_layers):
            values = linear(values, indices=indices)
            if self.use_layer_norm:
                values = self.layer_norms[layer_index](values, indices=indices)
            values = F.relu(values)
        values = self.output_layer(values, indices=indices)
        return values.squeeze(-1)


# -----------------------------------------------------------------------------
# RLPD learner
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
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def _make_critic_optimizer(
    critic: nn.Module,
    learning_rate: float,
    weight_decay: Optional[float],
) -> torch.optim.Optimizer:
    if weight_decay is None:
        return torch.optim.Adam(critic.parameters(), lr=learning_rate)

    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in critic.named_parameters():
        if name.endswith("bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


class RLPDLearner:
    def __init__(
        self,
        actor: Actor,
        critic: EnsembleCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temperature_learning_rate: float,
        init_temperature: float,
        target_entropy: float,
        discount: float,
        tau: float,
        num_min_qs: Optional[int],
        backup_entropy: bool,
        device: torch.device,
    ):
        self.actor = actor
        self.critic = critic
        self.target_critic = deepcopy(critic).to(device)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.log_temperature = torch.tensor(
            math.log(init_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=temperature_learning_rate
        )
        self.target_entropy = float(target_entropy)
        self.discount = float(discount)
        self.tau = float(tau)
        self.num_min_qs = num_min_qs
        self.backup_entropy = bool(backup_entropy)
        self.device = device

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _sample_target_indices(self) -> Optional[torch.Tensor]:
        if self.num_min_qs is None:
            return None
        if self.num_min_qs >= self.critic.ensemble_size:
            return None
        return torch.randperm(
            self.critic.ensemble_size, device=self.device
        )[: self.num_min_qs]

    def update_critic(self, batch: TorchBatch) -> Dict[str, float]:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                batch["next_observations"], reparameterize=True
            )
            indices = self._sample_target_indices()
            next_qs = self.target_critic(
                batch["next_observations"], next_actions, indices=indices
            )
            next_q = next_qs.min(dim=0).values
            target_q = (
                batch["rewards"]
                + self.discount * batch["masks"] * next_q
            )
            if self.backup_entropy:
                target_q = target_q - (
                    self.discount
                    * batch["masks"]
                    * self.temperature.detach()
                    * next_log_prob
                )

        predicted_qs = self.critic(batch["observations"], batch["actions"])
        critic_loss = (predicted_qs - target_q.unsqueeze(0)).pow(2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Official update_critic performs the target update immediately here.
        _soft_update(self.target_critic, self.critic, self.tau)

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "q": float(predicted_qs.detach().mean().item()),
        }

    def update_actor(self, batch: TorchBatch) -> Tuple[Dict[str, float], torch.Tensor]:
        actions, log_prob = self.actor.sample(
            batch["observations"], reparameterize=True
        )
        with _frozen_parameters(self.critic):
            q_values = self.critic(batch["observations"], actions).mean(dim=0)
        actor_loss = (
            self.temperature.detach() * log_prob - q_values
        ).mean()
        entropy = -log_prob.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return (
            {
                "actor_loss": float(actor_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
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
            "temperature_loss": float(temperature_loss.detach().item()),
        }

    def update(self, super_batch: TorchBatch, utd_ratio: int) -> Dict[str, float]:
        """Apply the asymmetric update schedule in RLPD Algorithm 1.

        The input super-batch contains ``batch_size * utd_ratio`` transitions
        and is split into ``utd_ratio`` contiguous mini-batches. The critic and
        target critic are updated on every mini-batch. After all critic updates,
        the actor and entropy temperature are each updated exactly once using
        the final mini-batch.

        Thus, per environment step:

            critic updates      = utd_ratio
            target updates      = utd_ratio
            actor updates       = 1
            temperature updates = 1

        This follows the loop placement in the paper pseudocode.
        """
        if utd_ratio < 1:
            raise ValueError("utd_ratio must be positive.")

        total_size = next(iter(super_batch.values())).shape[0]
        if total_size % utd_ratio != 0:
            raise ValueError(
                f"Super-batch size {total_size} is not divisible by UTD "
                f"ratio {utd_ratio}."
            )

        mini_batch_size = total_size // utd_ratio
        critic_info: Dict[str, float] = {}
        final_mini_batch: Optional[TorchBatch] = None

        for update_index in range(utd_ratio):
            start = update_index * mini_batch_size
            end = start + mini_batch_size
            mini_batch = {
                key: value[start:end] for key, value in super_batch.items()
            }

            critic_info = self.update_critic(mini_batch)
            final_mini_batch = mini_batch

        if final_mini_batch is None:
            raise RuntimeError("No mini-batch was produced for the RLPD update.")

        actor_info, entropy = self.update_actor(final_mini_batch)
        temperature_info = self.update_temperature(entropy)

        return {**actor_info, **critic_info, **temperature_info}


# -----------------------------------------------------------------------------
# Batch handling and evaluation
# -----------------------------------------------------------------------------


def _combine_interleaved(
    offline_batch: NumpyBatch,
    online_batch: NumpyBatch,
) -> NumpyBatch:
    if offline_batch.keys() != online_batch.keys():
        raise ValueError("Offline and online batches have different fields.")
    combined = {}
    for key in offline_batch:
        offline_value = np.asarray(offline_batch[key])
        online_value = np.asarray(online_batch[key])
        if offline_value.shape[0] != online_value.shape[0]:
            raise ValueError(
                "Official combine semantics require equal offline and online "
                "batch sizes."
            )
        output = np.empty(
            (
                offline_value.shape[0] + online_value.shape[0],
                *offline_value.shape[1:],
            ),
            dtype=offline_value.dtype,
        )
        output[0::2] = offline_value
        output[1::2] = online_value
        combined[key] = output
    return combined


def _shift_antmaze_rewards(batch: NumpyBatch, env_name: str) -> NumpyBatch:
    if "antmaze" not in env_name.lower():
        return batch
    shifted = dict(batch)
    shifted["rewards"] = np.asarray(batch["rewards"], dtype=np.float32) - 1.0
    return shifted


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


def _run_utd_updates(
    learner: RLPDLearner,
    super_batch: NumpyBatch,
    utd_ratio: int,
    device: torch.device,
) -> Dict[str, float]:
    # Convert once, then let the learner reproduce the official slicing and
    # update order internally.
    torch_super_batch = _to_torch_batch(super_batch, device)
    return learner.update(torch_super_batch, utd_ratio)


@torch.no_grad()
def evaluate(
    actor: Actor,
    env: gym.Env,
    device: torch.device,
    num_episodes: int,
) -> Dict[str, float]:
    returns = []
    lengths = []
    actor.eval()
    for _ in range(num_episodes):
        observation = _reset_env(env)
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = actor.eval_action(observation, device)
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


# -----------------------------------------------------------------------------
# Training
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


def _validate_config(config: TrainConfig) -> None:
    if config.batch_size < 1 or config.utd_ratio < 1:
        raise ValueError("batch_size and utd_ratio must be positive.")
    if config.num_critics < 1:
        raise ValueError("num_critics must be positive.")
    if config.num_min_qs is not None and not (
        1 <= config.num_min_qs <= config.num_critics
    ):
        raise ValueError("num_min_qs must be in [1, num_critics] or None.")
    if not math.isclose(config.offline_ratio, 0.5, abs_tol=1e-12):
        raise ValueError(
            "The official interleaving implementation requires offline_ratio=0.5."
        )
    if config.batch_size % 2 != 0:
        raise ValueError(
            "batch_size must be even so every UTD mini-batch is 50/50 mixed."
        )
    if config.start_training < 1:
        raise ValueError("start_training must be at least 1.")


def _create_offline_dataset(
    env: gym.Env,
    config: TrainConfig,
):
    source = config.dataset_source.lower()
    if source == "auto":
        source = "binary" if "binary" in config.env_name.lower() else "d4rl"
    if source == "d4rl":
        return D4RLDataset(env, seed=config.offline_dataset_seed)
    if source == "binary":
        if BinaryDataset is None:
            raise ImportError(
                "BinaryDataset is unavailable. Install the official RLPD "
                "repository and its Adroit dataset dependencies."
            )
        return ExternalDatasetAdapter(
            BinaryDataset(env, include_bc_data=config.binary_include_bc)
        )
    raise ValueError(f"Unknown dataset_source: {config.dataset_source}")


def train(config: TrainConfig) -> None:
    _validate_config(config)
    _set_seed(config.train_seed, config.deterministic_torch)
    device = torch.device(config.device)

    train_env = wrap_gym(gym.make(config.env_name), rescale_actions=True)
    # The official training script applies RecordEpisodeStatistics only to the
    # interaction environment, not to the evaluation environment.
    train_env = gym.wrappers.RecordEpisodeStatistics(train_env, deque_size=1)
    eval_env = wrap_gym(gym.make(config.env_name), rescale_actions=True)
    _seed_env(train_env, config.train_seed)
    _seed_env(eval_env, config.train_seed + config.eval_seed_offset)

    offline_dataset = _create_offline_dataset(train_env, config)

    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise TypeError("The wrapped observation space must be Box.")
    if not isinstance(train_env.action_space, gym.spaces.Box):
        raise TypeError("The wrapped action space must be Box.")
    observation_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))
    hidden_dims = tuple(config.hidden_dim for _ in range(config.hidden_layers))

    actor = Actor(observation_dim, action_dim, hidden_dims).to(device)
    critic = EnsembleCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        ensemble_size=config.num_critics,
        layer_norm=config.critic_layer_norm,
    ).to(device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.actor_learning_rate
    )
    critic_optimizer = _make_critic_optimizer(
        critic,
        learning_rate=config.critic_learning_rate,
        weight_decay=config.critic_weight_decay,
    )
    target_entropy = (
        -action_dim / 2.0
        if config.target_entropy is None
        else float(config.target_entropy)
    )
    learner = RLPDLearner(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        temperature_learning_rate=config.temperature_learning_rate,
        init_temperature=config.init_temperature,
        target_entropy=target_entropy,
        discount=config.discount,
        tau=config.tau,
        num_min_qs=config.num_min_qs,
        backup_entropy=config.backup_entropy,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        capacity=config.num_total_steps,
        seed=config.train_seed,
    )

    run_name = (
        f"{config.env_name}_seed{config.train_seed}"
    )
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
        f"critics={config.num_critics}, num_min_qs={config.num_min_qs}\n"
        f"UTD={config.utd_ratio}, batch_size={config.batch_size}\n"
        f"updates/env-step: critic={config.utd_ratio}, "
        f"actor=1, temperature=1\n"
        f"start_training={config.start_training}, steps={config.num_total_steps}\n"
        f"log_dir={log_dir}",
        flush=True,
    )

    super_batch_size = config.batch_size * config.utd_ratio
    half_super_batch = super_batch_size // 2

    for pretrain_step in trange(
        config.pretrain_steps, desc="RLPD offline pretraining"
    ):
        super_batch = offline_dataset.sample(super_batch_size)
        super_batch = _shift_antmaze_rewards(super_batch, config.env_name)
        metrics = _run_utd_updates(
            learner, super_batch, config.utd_ratio, device
        )
        if pretrain_step % config.log_every == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"offline-training/{key}", value, pretrain_step)
        if pretrain_step % config.eval_every == 0:
            offline_eval = evaluate(
                actor, eval_env, device, config.eval_episodes
            )
            writer.add_scalar(
                "offline-evaluation/return", offline_eval["return"], pretrain_step
            )
            writer.add_scalar(
                "offline-evaluation/length", offline_eval["length"], pretrain_step
            )
            print(
                f"[offline eval] step={pretrain_step}, "
                f"return={offline_eval['return']:.3f}, "
                f"length={offline_eval['length']:.1f}",
                flush=True,
            )
            writer.flush()

    observation = _reset_env(train_env)
    episode_return = 0.0
    episode_length = 0

    for step in trange(
        config.num_total_steps + 1,
        desc="RLPD online fine-tuning",
    ):
        if step < config.start_training:
            action = train_env.action_space.sample()
        else:
            action = actor.sample_action(observation, device)

        next_observation, reward, done, info = _step_env(train_env, action)
        # Match the released script, which checks for presence of the key.
        timeout = "TimeLimit.truncated" in info
        mask = 1.0 if (not done or timeout) else 0.0
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            mask=mask,
            done=done,
            next_observation=next_observation,
        )

        episode_return += reward
        episode_length += 1
        observation = next_observation
        if done:
            writer.add_scalar("training/reward", episode_return, step)
            writer.add_scalar("training/length", episode_length, step)
            observation = _reset_env(train_env)
            episode_return = 0.0
            episode_length = 0

        metrics: Dict[str, float] = {}
        if step >= config.start_training:
            online_batch = replay_buffer.sample(half_super_batch)
            offline_batch = offline_dataset.sample(half_super_batch)
            super_batch = _combine_interleaved(offline_batch, online_batch)
            super_batch = _shift_antmaze_rewards(super_batch, config.env_name)
            metrics = _run_utd_updates(
                learner, super_batch, config.utd_ratio, device
            )

        if metrics and step % config.log_every == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"training/{key}", value, step)

        if step % config.eval_every == 0:
            eval_info = evaluate(
                actor, eval_env, device, config.eval_episodes
            )
            writer.add_scalar("evaluation/reward", eval_info["return"], step)
            writer.add_scalar("evaluation/length", eval_info["length"], step)
            normalized_score = _get_normalized_score(
                eval_env, eval_info["return"]
            )
            message = (
                f"[eval] step={step}, return={eval_info['return']:.3f}, "
                f"length={eval_info['length']:.1f}"
            )
            if normalized_score is not None:
                writer.add_scalar(
                    "evaluation/d4rl_normalized_reward",
                    normalized_score,
                    step,
                )
                message += f", normalized_score={normalized_score:.3f}"
            print(message, flush=True)
            writer.flush()

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": learner.target_critic.state_dict(),
                "log_temperature": learner.log_temperature.detach().cpu(),
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
