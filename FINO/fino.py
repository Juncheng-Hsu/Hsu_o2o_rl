"""Single-file, state-based PyTorch implementation of FINO.

Reference: Shin et al., Flow Matching with Injected Noise for
Offline-to-Online Reinforcement Learning, ICLR 2026.
Author implementation: https://github.com/CTID282/FINO
Reference commit: de49317abaf6f86e258bba7ebee991c98f2e6360.

The noise schedule, candidate selection, GMM entropy approximation and beta
adaptation follow the RELEASED CODE, including documented paper/code differences.
Run: python fino.py --env_name=antmaze-umaze-v2 --train_seed=0
Use --print_config=True to inspect the resolved preset without loading data.
Unreported tasks have explicit transfer presets, not claimed tuned settings.
"""

# Portions implement the algorithm distributed under the following license:
# The MIT License (MIT)
# Copyright (c) 2025 FQL Authors
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NumpyBatch = Dict[str, np.ndarray]
TorchBatch = Dict[str, torch.Tensor]
ALGORITHM = "FINO"
REFERENCE_COMMIT = "de49317abaf6f86e258bba7ebee991c98f2e6360"
BATCH_KEYS = ("observations", "actions", "rewards", "masks", "next_observations")


@dataclass
class TrainConfig:
    env_name: str = "antmaze-umaze-v2"
    auto_env_config: bool = True
    strict_official_only: bool = False
    config_source: str = "unresolved"
    print_config: bool = False

    # None means use the environment preset. Explicit CLI values take precedence.
    alpha: Optional[float] = None
    discount: Optional[float] = None
    q_agg: Optional[str] = None
    normalize_q_loss: Optional[bool] = None
    offline_steps: Optional[int] = None
    online_steps: int = 1_000_000

    actor_hidden_dims: Tuple[int, ...] = (512, 512, 512, 512)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    critic_layer_norm: bool = True
    layer_norm_eps: float = 1e-6
    learning_rate: float = 3e-4
    tau: float = 0.005
    batch_size: int = 256
    flow_steps: int = 10
    # Numerical guard only; inactive at normal Q scales.
    normalization_eps: float = 1e-8

    # Released FINO settings. beta is an inverse sampling temperature, NOT
    # SAC's entropy coefficient. Entropy adaptation is online-only.
    noise_scale: float = 0.1
    initial_beta: float = 10.0
    beta_learning_rate: float = 0.1
    entropy_update_every: int = 50_000
    entropy_samples: int = 200
    entropy_components: int = 3
    entropy_chunk_size: int = 25

    replay_buffer_size: int = 1_500_000
    balanced_sampling: bool = False
    utd_ratio: int = 1
    action_clip_eps: float = 1e-5
    max_episode_steps: Optional[int] = None
    # Only this file's completed offline checkpoints may initialize fine-tuning.
    pretrained_checkpoint: Optional[str] = None

    train_seed: int = 0
    eval_seed_offset: int = 10_000
    deterministic_torch: bool = False
    eval_every: int = 5_000
    eval_episodes: int = 10
    log_every: int = 1_000
    save_every: int = 100_000
    checkpoints_path: Optional[str] = None
    log_root: str = "logs/FINO"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Automatic environment presets
# -----------------------------------------------------------------------------


LOCOMOTION_ENVS = {
    f"{agent}-{quality}-v2"
    for agent in ("halfcheetah", "hopper", "walker2d")
    for quality in ("random", "medium", "medium-replay", "medium-expert", "expert")
}
ANTMAZE_ENVS = {
    "antmaze-umaze-v2", "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2", "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2", "antmaze-large-diverse-v2",
}
ADROIT_ENVS = {
    f"{task}-{quality}-v1"
    for task in ("pen", "door", "hammer", "relocate")
    for quality in ("human", "cloned", "expert")
}
SUPPORTED_D4RL_ENVS = LOCOMOTION_ENVS | ANTMAZE_ENVS | ADROIT_ENVS

def _environment_preset(config: TrainConfig):
    name = config.env_name
    preset = dict(alpha=10.0, discount=0.99, q_agg="mean", normalize_q_loss=False)
    source = "fino_released_o2o_reference"
    if name in ANTMAZE_ENVS:
        preset["alpha"] = 3.0 if "large" in name else 10.0
    elif name in ADROIT_ENVS:
        task, quality, _ = name.split("-")
        preset["alpha"] = 10000.0 if task == "relocate" else 1000.0
        # FINO's released relocate command omits q_agg=min (unlike FQL).
        preset["q_agg"] = "mean" if task == "relocate" else "min"
        if quality != "cloned":
            source = "adapted_from_fino_cloned_o2o"
    elif name in LOCOMOTION_ENVS:
        # User-confirmed transfer rule, shared across dataset qualities.
        # Not a reported FINO tuning result.
        preset.update(alpha=1.0, normalize_q_loss=True)
        source = "adapted_locomotion_normalized_q_alpha1"
    else:
        raise ValueError(f"Unsupported D4RL environment: {name!r}. See the task list in README.md.")
    return preset, 1_000_000, source


def _resolve_config(config: TrainConfig) -> TrainConfig:
    config.env_name = config.env_name.lower()
    preset, offline_steps, source = _environment_preset(config)
    if config.strict_official_only and (source.startswith("adapted") or not config.auto_env_config):
        raise ValueError(f"No author-reported preset for this run: {source}.")
    if not config.auto_env_config:
        if config.alpha is None:
            raise ValueError("Set alpha explicitly when auto_env_config=False.")
        preset = dict(alpha=10.0, discount=0.99, q_agg="mean", normalize_q_loss=False)
        source = "custom"
    overrides = []
    for key, value in preset.items():
        if getattr(config, key) is None:
            setattr(config, key, value)
        elif getattr(config, key) != value:
            overrides.append(key)
    if overrides:
        source += ":override=" + ",".join(overrides)
    config.config_source = source
    if config.offline_steps is None:
        config.offline_steps = offline_steps
    if config.alpha < 0 or not math.isfinite(config.alpha):
        raise ValueError("alpha must be finite and nonnegative.")
    if not 0 <= config.discount < 1 or not 0 < config.tau <= 1:
        raise ValueError("Require 0 <= discount < 1 and 0 < tau <= 1.")
    if config.q_agg not in {"mean", "min"}:
        raise ValueError("q_agg must be 'mean' or 'min'.")
    for key in ("batch_size", "flow_steps", "utd_ratio", "replay_buffer_size", "eval_episodes", "log_every"):
        if getattr(config, key) <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in ("offline_steps", "online_steps", "eval_every", "save_every"):
        if getattr(config, key) < 0:
            raise ValueError(f"{key} must be nonnegative.")
    for key in ("learning_rate", "layer_norm_eps", "normalization_eps"):
        if not math.isfinite(getattr(config, key)) or getattr(config, key) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if config.balanced_sampling and config.batch_size % 2:
        raise ValueError("balanced_sampling requires an even batch_size.")
    if not 0 <= config.action_clip_eps < 1:
        raise ValueError("action_clip_eps must be in [0, 1).")
    if config.max_episode_steps is not None and config.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive.")
    for dims in (config.actor_hidden_dims, config.critic_hidden_dims):
        if not dims or any(d <= 0 for d in dims):
            raise ValueError("Hidden dimensions must be a nonempty sequence of positive integers.")
    if config.train_seed < 0 or config.eval_seed_offset < 0:
        raise ValueError("Seeds must be nonnegative.")
    for key in ("noise_scale", "initial_beta", "beta_learning_rate"):
        if not math.isfinite(getattr(config, key)) or getattr(config, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    for key in ("entropy_update_every", "entropy_samples", "entropy_components", "entropy_chunk_size"):
        if getattr(config, key) <= 0:
            raise ValueError(f"{key} must be positive.")
    if config.entropy_samples < max(2, config.entropy_components):
        raise ValueError("entropy_samples must be >= 2 and >= entropy_components.")
    return config


# -----------------------------------------------------------------------------
# Environments and replay. Heavy environment imports are deliberately lazy.
# -----------------------------------------------------------------------------


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def _reset_env(env, seed: Optional[int] = None):
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:  # Legacy Gym used by D4RL.
            env.seed(seed)
            result = env.reset()
        env.action_space.seed(seed)
    observation = result[0] if isinstance(result, tuple) else result
    return np.asarray(observation, dtype=np.float32)


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    else:
        observation, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated)
    return (
        np.asarray(observation, dtype=np.float32), float(reward),
        bool(terminated), bool(truncated), info,
    )


def _horizon(env, config: TrainConfig) -> int:
    if config.max_episode_steps is not None:
        return config.max_episode_steps
    spec = getattr(env, "spec", None)
    steps = getattr(spec, "max_episode_steps", None) or getattr(env, "_max_episode_steps", None)
    if steps is None:
        raise ValueError("Environment has no episode horizon; set max_episode_steps explicitly.")
    return int(steps)


def _training_reward(reward, env_name: str):
    return reward - 1.0 if env_name in ANTMAZE_ENVS else reward


def _prepare_dataset(raw: Mapping[str, np.ndarray], env_name: str, clip_eps: float) -> NumpyBatch:
    for key in ("observations", "actions", "rewards", "next_observations"):
        if key not in raw:
            raise ValueError(f"Dataset is missing {key!r}.")
    if "masks" in raw:
        masks = np.asarray(raw["masks"], dtype=np.float32).reshape(-1)
    elif "terminals" in raw:
        masks = 1.0 - np.asarray(raw["terminals"], dtype=np.float32).reshape(-1)
    else:
        raise ValueError("Dataset needs masks or terminals; timeouts must not disable bootstrapping.")
    arrays = {
        "observations": np.asarray(raw["observations"], dtype=np.float32),
        "actions": np.clip(np.asarray(raw["actions"], dtype=np.float32), -1 + clip_eps, 1 - clip_eps),
        "rewards": _training_reward(np.asarray(raw["rewards"], dtype=np.float32).reshape(-1), env_name),
        "masks": masks,
        "next_observations": np.asarray(raw["next_observations"], dtype=np.float32),
    }
    n = len(arrays["rewards"])
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError("Dataset arrays must have the same nonzero length.")
    if arrays["observations"].ndim != 2 or arrays["actions"].ndim != 2:
        raise ValueError("Only flat state observations and continuous action vectors are supported.")
    if arrays["next_observations"].shape != arrays["observations"].shape:
        raise ValueError("observations and next_observations have different shapes.")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Dataset contains NaN or infinite values.")
    if np.any((masks < 0) | (masks > 1)):
        raise ValueError("Bootstrap masks must lie in [0, 1].")
    return arrays


def _make_env_and_dataset(config: TrainConfig):
    import gym
    import d4rl

    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)
    raw = d4rl.qlearning_dataset(env)
    try:
        for instance in (env, eval_env):
            if len(instance.observation_space.shape) != 1 or len(instance.action_space.shape) != 1:
                raise ValueError("This single-file port requires vector observations and actions.")
            if not (np.allclose(instance.action_space.low, -1) and np.allclose(instance.action_space.high, 1)):
                raise ValueError("FQL/FINO reference policies require action bounds [-1, 1].")
        arrays = _prepare_dataset(raw, config.env_name, config.action_clip_eps)
        if arrays["observations"].shape[1:] != env.observation_space.shape:
            raise ValueError("Dataset observation shape does not match the environment.")
        if arrays["actions"].shape[1:] != env.action_space.shape:
            raise ValueError("Dataset action shape does not match the environment.")
    except Exception:
        env.close()
        eval_env.close()
        raise
    return env, eval_env, arrays


class ArrayDataset:
    def __init__(self, arrays: NumpyBatch, seed: int):
        self.arrays = arrays
        self.size = len(arrays["rewards"])
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> NumpyBatch:
        if self.size == 0:
            raise ValueError("Cannot sample an empty dataset.")
        indices = self.rng.integers(self.size, size=batch_size)
        return {key: self.arrays[key][indices] for key in BATCH_KEYS}


class ReplayBuffer(ArrayDataset):
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, seed: int):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        arrays = {
            "observations": np.empty((capacity, observation_dim), dtype=np.float32),
            "actions": np.empty((capacity, action_dim), dtype=np.float32),
            "rewards": np.empty(capacity, dtype=np.float32),
            "masks": np.empty(capacity, dtype=np.float32),
            "next_observations": np.empty((capacity, observation_dim), dtype=np.float32),
        }
        super().__init__(arrays, seed)
        self.capacity = capacity
        self.size = 0
        self.pointer = 0

    def initialize(self, arrays: NumpyBatch) -> None:
        n = len(arrays["rewards"])
        if self.size != 0 or n > self.capacity:
            raise ValueError("Initial dataset must fit into an empty replay buffer.")
        for key in BATCH_KEYS:
            self.arrays[key][:n] = arrays[key]
        self.size = n
        self.pointer = n % self.capacity

    def add(self, observation, action, reward, mask, next_observation) -> None:
        for key, value in zip(BATCH_KEYS, (observation, action, reward, mask, next_observation)):
            self.arrays[key][self.pointer] = value
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


def _to_torch(batch: NumpyBatch, device: torch.device) -> TorchBatch:
    return {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in batch.items()}


def _sample_online_batch(offline: ArrayDataset, replay: ReplayBuffer, config: TrainConfig):
    if not config.balanced_sampling:
        return replay.sample(config.batch_size)
    half = config.batch_size // 2
    left, right = offline.sample(half), replay.sample(half)
    return {key: np.concatenate((left[key], right[key]), axis=0) for key in BATCH_KEYS}


# -----------------------------------------------------------------------------
# Networks and FQL learning rules
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int,
                 layer_norm: bool, layer_norm_eps: float):
        super().__init__()
        layers = []
        dims = (input_dim,) + tuple(hidden_dims) + (output_dim,)
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(left, right)
            # Flax variance_scaling(1, 'fan_avg', 'uniform').
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(dims) - 2:
                # Match Flax's approximate GELU and activation -> LayerNorm order.
                layers.append(nn.GELU(approximate="tanh"))
                if layer_norm:
                    layers.append(nn.LayerNorm(right, eps=layer_norm_eps))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ActorVectorField(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, config: TrainConfig, time_conditioned: bool):
        super().__init__()
        self.time_conditioned = time_conditioned
        self.net = MLP(
            observation_dim + action_dim + int(time_conditioned),
            config.actor_hidden_dims, action_dim,
            config.actor_layer_norm, config.layer_norm_eps,
        )

    def forward(self, observations, actions, times=None):
        inputs = [observations, actions]
        if self.time_conditioned:
            if times is None:
                raise ValueError("The flow teacher requires time inputs.")
            inputs.append(times)
        return self.net(torch.cat(inputs, dim=-1))


class DoubleQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, config: TrainConfig):
        super().__init__()
        self.qs = nn.ModuleList([
            MLP(observation_dim + action_dim, config.critic_hidden_dims, 1,
                config.critic_layer_norm, config.layer_norm_eps)
            for _ in range(2)
        ])

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), dim=-1)
        return torch.stack([q(inputs).squeeze(-1) for q in self.qs], dim=0)


@contextmanager
def _frozen_parameters(module: nn.Module):
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


class FINOLearner:
    def __init__(self, observation_dim: int, action_dim: int, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.critic = DoubleQ(observation_dim, action_dim, config).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_bc_flow = ActorVectorField(observation_dim, action_dim, config, True).to(self.device)
        self.actor_onestep_flow = ActorVectorField(observation_dim, action_dim, config, False).to(self.device)
        # A single Adam matches the released joint loss/optimizer. Target Q is
        # excluded, and actor Q gradients never accumulate into critic weights.
        self.optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.actor_bc_flow.parameters())
            + list(self.actor_onestep_flow.parameters()), lr=config.learning_rate,
        )
        self.train_rng = torch.Generator(device=self.device).manual_seed(config.train_seed + 1)
        self.action_rng = torch.Generator(device=self.device).manual_seed(config.train_seed + 2)
        self.updates = 0
        self.beta = config.initial_beta
        self.entropy_updates = 0

    @torch.no_grad()
    def compute_flow_actions(self, observations, noises):
        actions = noises
        for step in range(self.config.flow_steps):
            times = torch.full((*actions.shape[:-1], 1), step / self.config.flow_steps, device=self.device)
            actions = actions + self.actor_bc_flow(observations, actions, times) / self.config.flow_steps
        # Clip only the final Euler solution, not intermediate ODE states.
        return actions.clamp(-1, 1)

    @torch.no_grad()
    def sample_batch_actions(self, observations, generator=None):
        if generator is None:
            generator = self.train_rng
        noises = torch.randn((*observations.shape[:-1], self.action_dim), device=self.device, generator=generator)
        return self.actor_onestep_flow(observations, noises).clamp(-1, 1)

    @torch.no_grad()
    def sample_actions(self, observations, evaluation: bool = False, generator=None):
        if generator is None:
            generator = self.action_rng
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        single = observations.ndim == 1
        if single:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2:
            raise ValueError("Expected one state vector or a batch of state vectors.")
        actions = self._sample_selected_actions(observations, generator, evaluation=evaluation)[0]
        # Keep the action dimension even when action_dim=1; do not use squeeze().
        return (actions[0] if single else actions).cpu().numpy()

    @torch.no_grad()
    def _sample_selected_actions(self, observations, generator, evaluation=False, replicas=1):
        """Return [replicas, batch, action_dim] selected candidate actions.

        Each entropy replica uses the same Q normalization as one call to the
        released sampler: mean absolute Q over all candidates AND batch states.
        """
        batch_size = len(observations)
        candidates = min(10, (self.action_dim + 1) // 2)
        noises = torch.randn((replicas, candidates, batch_size, self.action_dim),
                             device=self.device, generator=generator)
        states = observations[None, None].expand(replicas, candidates, -1, -1)
        actions = self.actor_onestep_flow(states, noises).clamp(-1, 1)
        qs = self.critic(states, actions)
        q = qs.min(dim=0).values if self.config.q_agg == "min" else qs.mean(dim=0)
        if evaluation:
            indices = q.argmax(dim=1)
        else:
            scale = q.abs().mean(dim=(1, 2), keepdim=True).clamp_min(self.config.normalization_eps)
            logits = (self.beta * q / scale).transpose(1, 2)
            probabilities = logits.softmax(dim=-1).reshape(-1, candidates)
            indices = torch.multinomial(probabilities, 1, generator=generator).reshape(replicas, batch_size)
        ordered = actions.permute(0, 2, 1, 3)
        return ordered.gather(2, indices[:, :, None, None].expand(-1, -1, 1, self.action_dim)).squeeze(2)

    def _flow_training_pair(self, actions, noise, times):
        interpolated = (1 - times) * noise + times * actions
        extra_noise = torch.randn(actions.shape, device=self.device, generator=self.train_rng)
        # Exact released implementation: sigma(t) = 0.1 * exp(10 * (t - 1)).
        # Its target remains a - z. Do not silently replace it by the different
        # interpolant/velocity-target formula in the paper.
        scale = self.config.noise_scale * torch.exp(10 * (times - 1))
        return interpolated + scale * extra_noise, actions - noise

    @torch.no_grad()
    def update_entropy(self, observations: torch.Tensor) -> Dict[str, float]:
        # The reference resets the JAX sampling key to train_seed each call.
        generator = torch.Generator(device=self.device).manual_seed(self.config.train_seed)
        chunks = []
        for start in range(0, self.config.entropy_samples, self.config.entropy_chunk_size):
            replicas = min(self.config.entropy_chunk_size, self.config.entropy_samples - start)
            selected = self._sample_selected_actions(observations, generator, replicas=replicas)
            chunks.append(selected.cpu().numpy())
        samples = np.concatenate(chunks, axis=0).transpose(1, 0, 2)
        # GMM random states are explicit for reproducibility. The released code
        # instead draws these from NumPy's shared global random stream.
        entropy = estimate_entropy_sklearn(
            samples, self.config.entropy_components, self.config.train_seed + self.entropy_updates,
        )
        target_entropy = -float(self.action_dim)
        self.beta = max(0.0, self.beta - self.config.beta_learning_rate * (target_entropy - entropy))
        self.entropy_updates += 1
        return {"entropy_estimate": entropy, "beta": self.beta}

    def _critic_loss(self, batch):
        with torch.no_grad():
            # Use the current one-step policy, not the teacher or a target actor.
            next_actions = self.sample_batch_actions(batch["next_observations"])
            next_qs = self.target_critic(batch["next_observations"], next_actions)
            next_q = next_qs.min(dim=0).values if self.config.q_agg == "min" else next_qs.mean(dim=0)
            target = batch["rewards"] + self.config.discount * batch["masks"] * next_q
        qs = self.critic(batch["observations"], batch["actions"])
        loss = (qs - target.unsqueeze(0)).square().mean()
        return loss, {"critic/loss": loss.detach(), "critic/q_mean": qs.detach().mean()}

    def _actor_loss(self, batch):
        observations, actions = batch["observations"], batch["actions"]
        noise = torch.randn(actions.shape, device=self.device, generator=self.train_rng)
        times = torch.rand((len(actions), 1), device=self.device, generator=self.train_rng)
        flow_inputs, target_velocity = self._flow_training_pair(actions, noise, times)
        velocity = self.actor_bc_flow(observations, flow_inputs, times)
        flow_loss = F.mse_loss(velocity, target_velocity)

        noise = torch.randn(actions.shape, device=self.device, generator=self.train_rng)
        teacher_actions = self.compute_flow_actions(observations, noise)
        student_actions = self.actor_onestep_flow(observations, noise)
        # Distill raw student outputs. Clipping happens only before querying Q.
        distill_loss = F.mse_loss(student_actions, teacher_actions)
        with _frozen_parameters(self.critic):
            q = self.critic(observations, student_actions.clamp(-1, 1)).mean(dim=0)
        q_loss = -q.mean()
        if self.config.normalize_q_loss:
            q_loss = q_loss / q.detach().abs().mean().clamp_min(self.config.normalization_eps)
        loss = flow_loss + self.config.alpha * distill_loss + q_loss
        return loss, {
            "actor/loss": loss.detach(), "actor/flow_loss": flow_loss.detach(),
            "actor/distill_loss": distill_loss.detach(), "actor/q_loss": q_loss.detach(),
            "actor/q_mean": q.detach().mean(),
        }

    def update(self, batch: TorchBatch) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        # Both losses are computed at the SAME pre-update parameters.
        critic_loss, critic_info = self._critic_loss(batch)
        actor_loss, actor_info = self._actor_loss(batch)
        loss = critic_loss + actor_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite FQL/FINO loss; inspect data and configuration.")
        loss.backward()
        with torch.no_grad():
            # Important: the released target_update reads self.network.params,
            # i.e. the OLD critic, not new_network.params. Preserve that ordering.
            for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(current, self.config.tau)
        self.optimizer.step()
        self.updates += 1
        return {key: float(value.cpu()) for key, value in {**critic_info, **actor_info}.items()}

    def state_dict(self):
        return {
            "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
            "actor_bc_flow": self.actor_bc_flow.state_dict(),
            "actor_onestep_flow": self.actor_onestep_flow.state_dict(),
            "optimizer": self.optimizer.state_dict(), "updates": self.updates,
            "train_rng": self.train_rng.get_state(), "action_rng": self.action_rng.get_state(),
            "beta": self.beta, "entropy_updates": self.entropy_updates,
        }

    def load_state_dict(self, state):
        for name in ("critic", "target_critic", "actor_bc_flow", "actor_onestep_flow"):
            getattr(self, name).load_state_dict(state[name])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.train_rng.set_state(state["train_rng"].cpu())
        self.action_rng.set_state(state["action_rng"].cpu())
        self.beta = float(state["beta"])
        self.entropy_updates = int(state["entropy_updates"])


def estimate_entropy_sklearn(actions: np.ndarray, num_components: int = 3, seed: int = 0) -> float:
    """Released GMM approximation H(weights) + sum_k weights_k H(Gaussian_k).

    This is the mixture-entropy upper-bound approximation used by FINO, not
    exact mixture entropy or the discrete entropy of candidate probabilities.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("FINO entropy adaptation requires scikit-learn; see README.md.") from exc
    if actions.ndim != 3 or actions.shape[1] < max(2, num_components) or not np.isfinite(actions).all():
        raise ValueError("GMM input must be finite [batch, samples, action_dim] with enough samples.")
    entropies = []
    dimension = actions.shape[-1]
    for index, samples in enumerate(actions):
        model = GaussianMixture(n_components=num_components, covariance_type="full",
                                random_state=(seed + index) % (2 ** 32))
        model.fit(samples)
        sign, logdet = np.linalg.slogdet(model.covariances_)
        if np.any(sign <= 0):
            raise FloatingPointError("GMM produced a non-positive covariance determinant.")
        component_entropy = 0.5 * dimension * (1 + np.log(2 * np.pi)) + 0.5 * logdet
        value = -(model.weights_ * np.log(model.weights_ + 1e-8)).sum()
        value += (model.weights_ * component_entropy).sum()
        entropies.append(value)
    result = float(np.mean(entropies))
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite GMM entropy estimate.")
    return result


# -----------------------------------------------------------------------------
# Evaluation, checkpoints and the standalone offline-to-online training loop
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(learner: FINOLearner, env, config: TrainConfig) -> Dict[str, float]:
    # Evaluation must not consume training/action RNG streams or replay samples.
    seed = config.train_seed + config.eval_seed_offset
    generator = torch.Generator(device=learner.device).manual_seed(seed)
    returns, lengths, successes = [], [], []
    for episode in range(config.eval_episodes):
        observation = _reset_env(env, seed + episode)
        episode_return, success, has_success = 0.0, 0.0, False
        for length in range(1, _horizon(env, config) + 1):
            action = learner.sample_actions(observation, evaluation=True, generator=generator)
            observation, reward, terminated, truncated, info = _step_env(env, action)
            episode_return += reward  # Raw environment reward, never training-shifted.
            if "success" in info:
                success = max(success, float(np.asarray(info["success"]).max()))
                has_success = True
            elif "goal_achieved" in info:
                success = max(success, float(np.asarray(info["goal_achieved"]).max()))
                has_success = True
            if terminated or truncated:
                break
        returns.append(episode_return)
        lengths.append(length)
        if has_success:
            successes.append(success)
    metrics = {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
               "episode_length": float(np.mean(lengths))}
    score_env = getattr(env, "unwrapped", env)
    if hasattr(score_env, "get_normalized_score"):
        metrics["normalized_score"] = float(score_env.get_normalized_score(np.mean(returns)) * 100.0)
    if successes:
        metrics["success_rate"] = float(np.mean(successes))
    return metrics


def _log_metrics(writer, metrics: Mapping[str, float], step: int, prefix: str) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def _save_checkpoint(learner, config, directory: Optional[Path], phase: str,
                     offline_steps: int, online_steps: int) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": ALGORITHM, "format_version": 1, "reference_commit": REFERENCE_COMMIT,
        "config": asdict(config), "learner": learner.state_dict(), "phase": phase,
        "offline_steps": offline_steps, "online_steps": online_steps,
        "observation_dim": learner.observation_dim, "action_dim": learner.action_dim,
    }
    name = "offline_final.pt" if phase == "offline_final" else f"{phase}_{online_steps if online_steps else offline_steps}.pt"
    path = directory / name
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_pretrained(learner, config: TrainConfig) -> int:
    # Load only files produced by this implementation; no JAX checkpoint conversion.
    checkpoint = torch.load(config.pretrained_checkpoint, map_location=learner.device, weights_only=False)
    if checkpoint.get("algorithm") != ALGORITHM or checkpoint.get("format_version") != 1:
        raise ValueError("Checkpoint algorithm or format does not match this implementation.")
    if checkpoint.get("phase") != "offline_final" or checkpoint.get("online_steps") != 0:
        raise ValueError("pretrained_checkpoint requires offline_final.pt; online replay is not saved.")
    if (checkpoint["observation_dim"], checkpoint["action_dim"]) != (learner.observation_dim, learner.action_dim):
        raise ValueError("Checkpoint observation/action dimensions do not match.")
    saved = checkpoint["config"]
    keys = (
        "env_name", "actor_hidden_dims", "critic_hidden_dims", "actor_layer_norm", "critic_layer_norm",
        "layer_norm_eps", "alpha", "discount", "q_agg", "normalize_q_loss", "flow_steps",
        "learning_rate", "tau", "action_clip_eps",
        "noise_scale",
    )
    for key in keys:
        if saved[key] != getattr(config, key):
            raise ValueError(f"Checkpoint configuration differs for {key}: {saved[key]!r} vs {getattr(config, key)!r}.")
    learner.load_state_dict(checkpoint["learner"])
    return int(checkpoint["offline_steps"])


def train(config: TrainConfig) -> None:
    config = _resolve_config(config)
    if config.print_config:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        return
    if config.online_steps > 0:
        # Fail before dataset loading/pretraining, rather than 50k online steps later.
        try:
            from sklearn.mixture import GaussianMixture  # noqa: F401
        except ImportError as exc:
            raise ImportError("FINO online training requires scikit-learn. See README.md for compatible versions.") from exc
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import trange

    _set_seed(config.train_seed, config.deterministic_torch)
    env, eval_env, arrays = _make_env_and_dataset(config)
    writer = None
    try:
        run_name = f"{config.env_name}/seed_{config.train_seed}_{time.time_ns()}"
        log_dir = Path(config.log_root) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({**asdict(config), "reference_commit": REFERENCE_COMMIT}, handle, indent=2)
        writer = SummaryWriter(str(log_dir))
        checkpoint_dir = Path(config.checkpoints_path) / run_name if config.checkpoints_path else None
        learner = FINOLearner(arrays["observations"].shape[1], arrays["actions"].shape[1], config)
        offline = ArrayDataset(arrays, config.train_seed + 3)
        print(f"[{ALGORITHM}] preset={config.config_source} alpha={config.alpha} "
              f"q_agg={config.q_agg} gamma={config.discount} offline_size={offline.size}", flush=True)
        if config.config_source.startswith("adapted"):
            print("[preset] Fixed task transfer; this is not an author-tuned configuration.", flush=True)

        completed_offline = 0
        if config.pretrained_checkpoint:
            completed_offline = _load_pretrained(learner, config)
            print(f"[checkpoint] Loaded offline initialization after {completed_offline} updates.", flush=True)
        else:
            for step in trange(1, config.offline_steps + 1, desc=f"{ALGORITHM} offline"):
                metrics = learner.update(_to_torch(offline.sample(config.batch_size), learner.device))
                completed_offline = step
                if step % config.log_every == 0 or step == config.offline_steps:
                    _log_metrics(writer, metrics, step, "offline")
                if config.eval_every and (step % config.eval_every == 0 or step == config.offline_steps):
                    evaluation = evaluate(learner, eval_env, config)
                    _log_metrics(writer, evaluation, step, "offline_evaluation")
                    print(f"[offline eval] updates={step}: {evaluation}", flush=True)
                if config.save_every and step % config.save_every == 0:
                    _save_checkpoint(learner, config, checkpoint_dir, "offline", step, 0)
            _save_checkpoint(learner, config, checkpoint_dir, "offline_final", completed_offline, 0)

        if config.online_steps == 0:
            return
        if config.eval_every:
            _log_metrics(writer, evaluate(learner, eval_env, config), 0, "evaluation")
        # Default author protocol: a single uniform buffer prefilled offline.
        # The optional balanced variant uses a separate, initially empty buffer.
        capacity = config.replay_buffer_size if config.balanced_sampling else max(config.replay_buffer_size, offline.size + 1)
        replay = ReplayBuffer(capacity, learner.observation_dim, learner.action_dim, config.train_seed + 4)
        if not config.balanced_sampling:
            replay.initialize(arrays)
            offline = replay  # Release the extra immutable copy.
        del arrays
        observation = _reset_env(env, config.train_seed)
        episode_return, episode_length = 0.0, 0
        for online_step in trange(1, config.online_steps + 1, desc=f"{ALGORITHM} online"):
            action = learner.sample_actions(observation)
            next_observation, reward, terminated, truncated, info = _step_env(env, action)
            episode_length += 1
            if episode_length >= _horizon(env, config) and not terminated:
                truncated = True
            replay.add(observation, action, _training_reward(reward, config.env_name),
                       float(not terminated), next_observation)
            episode_return += reward
            observation = next_observation
            if terminated or truncated:
                _log_metrics(writer, {"return": episode_return, "length": episode_length}, online_step, "exploration")
                observation = _reset_env(env)
                episode_return, episode_length = 0.0, 0
            # No random-action warmup, SAC entropy term, or added action noise.
            for _ in range(config.utd_ratio):
                batch = _to_torch(_sample_online_batch(offline, replay, config), learner.device)
                metrics = learner.update(batch)
            if online_step % config.log_every == 0 or online_step == config.online_steps:
                metrics["gradient_updates"] = learner.updates
                _log_metrics(writer, metrics, online_step, "online")
            if config.eval_every and (online_step % config.eval_every == 0 or online_step == config.online_steps):
                evaluation = evaluate(learner, eval_env, config)
                _log_metrics(writer, evaluation, online_step, "evaluation")
                print(f"[online eval] interactions={online_step}: {evaluation}", flush=True)
                writer.flush()
            # In the reference this is coupled to total-step evaluation ticks.
            # Preserve the default 50k schedule, but keep it independent of this
            # project's configurable evaluation frequency.
            if (completed_offline + online_step) % config.entropy_update_every == 0:
                entropy_metrics = learner.update_entropy(batch["observations"])
                _log_metrics(writer, entropy_metrics, online_step, "online_sampling")
            if config.save_every and online_step % config.save_every == 0:
                _save_checkpoint(learner, config, checkpoint_dir, "online", completed_offline, online_step)
        _save_checkpoint(learner, config, checkpoint_dir, "online_final", completed_offline, config.online_steps)
    finally:
        if writer is not None:
            writer.close()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    import pyrallis

    train(pyrallis.parse(config_class=TrainConfig))
