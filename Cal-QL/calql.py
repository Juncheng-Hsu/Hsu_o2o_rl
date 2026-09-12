"""Cal-QL in one PyTorch file for the RLPD-matched D4RL benchmark.

Algorithm source of truth:
  - Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for
    Efficient Online Fine-Tuning", NeurIPS 2023.
  - The authors' JAX/Flax repository: https://github.com/nakamotoo/Cal-QL

The user's Hsu_o2o_rl repository is used only as a style/CLI/logging reference.
No algorithmic choice is inherited from that repository.

The Cal-QL update follows the released implementation: SAC + CQL, with the
policy-sampled Q-values inside the conservative regularizer lower-bounded by a
reference-policy value. AntMaze uses behavior-trajectory Monte-Carlo return-to-
go exactly as in the released code. For D4RL locomotion, Appendix D explicitly
requires a fitted SARSA Q-function because the dataset does not terminate; this
file implements that missing reference-estimation stage. Standard D4RL Adroit
(human/cloned/expert) was not the paper's Adroit-Binary benchmark, so this file
uses the same paper-supported fitted-SARSA reference estimator while retaining
the Cal-QL objective and Adroit CQL/network settings. Dense D4RL rewards are
not replaced by the Adroit-Binary reward transform.

Supported directly: the 30 RLPD-matched tasks used by this project: 12 MuJoCo
locomotion, 12 standard D4RL Adroit, and 6 AntMaze tasks. Environment-dependent
algorithm parameters are selected from env_name; experiment-protocol parameters
(online budget, evaluation cadence, seeds, UTD, etc.) remain explicit CLI
arguments.
"""

import math
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    """Configuration for the single-file Cal-QL reproduction.

    Parameters under "experiment protocol" are intentionally NOT changed by the
    environment preset. They are the quantities that should be kept identical
    across baselines in a paper (online budget, evaluation cadence, seeds, etc.).
    Environment-dependent Cal-QL choices are resolved from ``env_name`` below.
    """

    # ------------------------------------------------------------------
    # Experiment protocol: set these yourself for the paper.
    # ------------------------------------------------------------------
    env_name: str = "antmaze-umaze-v2"
    train_seed: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Offline pre-training gradient updates and online environment interactions.
    # ``pretrain_steps=None`` selects the paper/released-code budget by domain:
    # 1M for AntMaze/locomotion and 20K for the Adroit family.  The standard
    # D4RL Adroit tasks supported here are a benchmark adaptation of the
    # paper's Adroit-Binary setup, so they inherit its 20K offline budget.
    pretrain_steps: Optional[int] = None
    num_total_steps: int = 1_000_000
    batch_size: int = 256
    online_utd_ratio: int = 1
    mixing_ratio: float = 0.5
    replay_buffer_size: int = 1_000_000

    eval_every: int = 5_000
    eval_episodes: int = 10
    log_every: int = 1_000
    checkpoints_path: Optional[str] = None
    log_root: str = "logs/CalQL"

    # The locomotion appendix states that the reference values are estimated by
    # fitting a SARSA Q-function, but the authors did not release its fitting
    # hyperparameters. Therefore these are explicit experimental parameters,
    # rather than silently pretending to be official hyperparameters.
    reference_train_steps: int = 1_000_000
    reference_batch_size: int = 256
    reference_learning_rate: float = 3e-4
    reference_tau: float = 5e-3
    reference_target_update_period: int = 1

    # ------------------------------------------------------------------
    # Core SAC/CQL settings. Defaults are from the author-released code.
    # Environment-dependent values marked Optional are filled automatically.
    # ------------------------------------------------------------------
    discount: float = 0.99
    tau: float = 5e-3
    policy_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 1e-4
    cql_lagrange_learning_rate: Optional[float] = None

    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    critic_hidden_dims: Optional[Tuple[int, ...]] = None
    reference_hidden_dims: Optional[Tuple[int, ...]] = None
    orthogonal_init: bool = True

    actor_log_std_min: float = -20.0
    actor_log_std_max: float = 2.0
    actor_log_std_multiplier_init: float = 1.0
    actor_log_std_offset_init: float = -1.0
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None

    cql_min_q_weight: Optional[float] = None
    cql_temperature: float = 1.0
    cql_n_actions: int = 10
    cql_importance_sample: bool = True
    cql_clip_diff_min: float = -float("inf")
    cql_clip_diff_max: float = float("inf")
    cql_lagrange: Optional[bool] = None
    cql_target_action_gap: Optional[float] = None
    cql_alpha_prime_init_log: float = 1.0
    cql_alpha_prime_max: float = 1_000_000.0
    enable_calql: bool = True

    max_target_backup: Optional[bool] = None
    backup_entropy: bool = False
    target_backup_n_actions: int = 10

    reward_scale: Optional[float] = None
    reward_bias: Optional[float] = None
    sparse_failure_reward: Optional[float] = None

    # Reference estimator: "mc" or "sarsa". "auto" selects by environment.
    # AntMaze follows the released MC-return implementation. D4RL locomotion
    # follows Appendix D and uses a fitted SARSA reference Q. Standard D4RL
    # Adroit is not an original Cal-QL benchmark; we use the same fitted-SARSA
    # mechanism so the Cal-QL objective itself is unchanged.
    reference_mode: str = "auto"

    online_use_cql: bool = True
    cql_min_q_weight_online: Optional[float] = None
    trajectories_per_iteration: int = 1

    clip_action: float = 0.99999
    max_episode_steps: Optional[int] = None
    eval_seed_offset: int = 42
    dataset_seed: Optional[int] = None
    deterministic_torch: bool = False


RLPD_MATCHED_ENVS = (
    # D4RL MuJoCo locomotion (12)
    "halfcheetah-random-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-random-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-random-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    # D4RL Adroit (12)
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1",
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",
    # D4RL AntMaze (6)
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
)


# -----------------------------------------------------------------------------
# Environment presets
# -----------------------------------------------------------------------------


def _is_antmaze(env_name: str) -> bool:
    return "antmaze" in env_name.lower()


def _is_locomotion(env_name: str) -> bool:
    name = env_name.lower()
    return any(token in name for token in ("halfcheetah", "hopper", "walker2d"))


def _is_standard_adroit(env_name: str) -> bool:
    name = env_name.lower()
    return any(token in name for token in ("pen-", "door-", "hammer-", "relocate-"))


def _resolve_environment_preset(config: TrainConfig) -> None:
    """Resolve only environment-dependent algorithm parameters.

    Source hierarchy:
      1) Cal-QL paper / appendix;
      2) author-released JAX code and run scripts;
      3) where the original work gives no setting for the requested benchmark,
         retain the Cal-QL objective and use the closest paper-supported domain
         setting, with the deviation documented in this file.
    """
    name = config.env_name.lower()

    if _is_antmaze(name):
        # Exact released AntMaze path: run_antmaze.sh + ConservativeSAC defaults.
        config.actor_hidden_dims = config.actor_hidden_dims or (256, 256)
        config.critic_hidden_dims = config.critic_hidden_dims or (256, 256, 256, 256)
        config.reference_hidden_dims = config.reference_hidden_dims or config.critic_hidden_dims
        config.cql_min_q_weight = 5.0 if config.cql_min_q_weight is None else config.cql_min_q_weight
        config.cql_lagrange = True if config.cql_lagrange is None else config.cql_lagrange
        config.cql_target_action_gap = 0.8 if config.cql_target_action_gap is None else config.cql_target_action_gap
        config.max_target_backup = True if config.max_target_backup is None else config.max_target_backup
        config.reward_scale = 10.0 if config.reward_scale is None else config.reward_scale
        config.reward_bias = -5.0 if config.reward_bias is None else config.reward_bias
        config.sparse_failure_reward = float(config.reward_bias)
        if config.reference_mode == "auto":
            config.reference_mode = "mc"
        if config.pretrain_steps is None:
            config.pretrain_steps = 1_000_000

    elif _is_locomotion(name):
        # Appendix D: Cal-QL uses a fitted SARSA Q-function to estimate the
        # reference values because locomotion datasets do not terminate normally.
        # Locomotion-specific CQL hyperparameters were not released; we therefore
        # use the author's ConservativeSAC/main defaults rather than inventing a
        # tuned per-task table.
        config.actor_hidden_dims = config.actor_hidden_dims or (256, 256)
        config.critic_hidden_dims = config.critic_hidden_dims or (256, 256)
        config.reference_hidden_dims = config.reference_hidden_dims or (256, 256)
        config.cql_min_q_weight = 5.0 if config.cql_min_q_weight is None else config.cql_min_q_weight
        config.cql_lagrange = False if config.cql_lagrange is None else config.cql_lagrange
        config.cql_target_action_gap = 1.0 if config.cql_target_action_gap is None else config.cql_target_action_gap
        config.max_target_backup = True if config.max_target_backup is None else config.max_target_backup
        config.reward_scale = 1.0 if config.reward_scale is None else config.reward_scale
        config.reward_bias = 0.0 if config.reward_bias is None else config.reward_bias
        config.sparse_failure_reward = None
        if config.reference_mode == "auto":
            config.reference_mode = "sarsa"
        if config.pretrain_steps is None:
            config.pretrain_steps = 1_000_000

    elif _is_standard_adroit(name):
        # The paper's Adroit experiments use *binary* tasks and custom demo data,
        # whereas the RLPD benchmark uses the standard D4RL human/cloned/expert
        # datasets. To support the RLPD-matched benchmark without changing the
        # Cal-QL objective, use the paper's Adroit CQL/network choices and a
        # fitted-SARSA reference estimator (the paper-supported fallback when a
        # reliable trajectory return is unavailable). Dense D4RL rewards are kept
        # unchanged; the +5/×10 transform is specific to Adroit-Binary rewards.
        config.actor_hidden_dims = config.actor_hidden_dims or (512, 512)
        config.critic_hidden_dims = config.critic_hidden_dims or (512, 512, 512)
        config.reference_hidden_dims = config.reference_hidden_dims or (512, 512, 512)
        config.cql_min_q_weight = 1.0 if config.cql_min_q_weight is None else config.cql_min_q_weight
        config.cql_lagrange = False if config.cql_lagrange is None else config.cql_lagrange
        config.cql_target_action_gap = 1.0 if config.cql_target_action_gap is None else config.cql_target_action_gap
        config.max_target_backup = True if config.max_target_backup is None else config.max_target_backup
        config.reward_scale = 1.0 if config.reward_scale is None else config.reward_scale
        config.reward_bias = 0.0 if config.reward_bias is None else config.reward_bias
        config.sparse_failure_reward = None
        if config.reference_mode == "auto":
            config.reference_mode = "sarsa"
        if config.pretrain_steps is None:
            config.pretrain_steps = 20_000

    else:
        raise ValueError(
            f"Unsupported environment '{config.env_name}'. This file is designed "
            "for the RLPD-matched D4RL MuJoCo/Adroit/AntMaze benchmark."
        )


def _validate_config(config: TrainConfig) -> None:
    _resolve_environment_preset(config)

    if config.reference_mode not in {"mc", "sarsa"}:
        raise ValueError("reference_mode must resolve to 'mc' or 'sarsa'.")
    required = {
        "actor_hidden_dims": config.actor_hidden_dims,
        "critic_hidden_dims": config.critic_hidden_dims,
        "reference_hidden_dims": config.reference_hidden_dims,
        "cql_min_q_weight": config.cql_min_q_weight,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "max_target_backup": config.max_target_backup,
        "reward_scale": config.reward_scale,
        "reward_bias": config.reward_bias,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"Environment preset did not resolve: {missing}")

    positive_ints = {
        "pretrain_steps": config.pretrain_steps,
        "num_total_steps": config.num_total_steps,
        "batch_size": config.batch_size,
        "online_utd_ratio": config.online_utd_ratio,
        "replay_buffer_size": config.replay_buffer_size,
        "eval_every": config.eval_every,
        "eval_episodes": config.eval_episodes,
        "reference_train_steps": config.reference_train_steps,
        "reference_batch_size": config.reference_batch_size,
        "reference_target_update_period": config.reference_target_update_period,
    }
    for key, value in positive_ints.items():
        if int(value) < 1:
            raise ValueError(f"{key} must be positive.")
    if not 0.0 <= config.mixing_ratio <= 1.0:
        raise ValueError("mixing_ratio must lie in [0, 1].")
    if not 0.0 < config.discount <= 1.0:
        raise ValueError("discount must lie in (0, 1].")
    if not 0.0 < config.tau <= 1.0:
        raise ValueError("tau must lie in (0, 1].")
    if not 0.0 < config.reference_tau <= 1.0:
        raise ValueError("reference_tau must lie in (0, 1].")
    if config.cql_temperature <= 0.0:
        raise ValueError("cql_temperature must be positive.")
    if not 0.0 < config.clip_action < 1.0:
        raise ValueError("clip_action must lie in (0, 1).")


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
        result = None
        try:
            result = self.env.seed(seed)
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
        return result


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


def _max_episode_steps(env: gym.Env, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    candidate = env
    for _ in range(16):
        if hasattr(candidate, "_max_episode_steps"):
            return int(candidate._max_episode_steps)
        if hasattr(candidate, "spec") and candidate.spec is not None:
            value = getattr(candidate.spec, "max_episode_steps", None)
            if value is not None:
                return int(value)
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    return 1_000


# -----------------------------------------------------------------------------
# Dataset and replay buffers
# -----------------------------------------------------------------------------


def _discounted_return_to_go(
    rewards: np.ndarray,
    terminals: np.ndarray,
    discount: float,
    sparse_failure_reward: Optional[float],
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
    if rewards.shape != terminals.shape:
        raise ValueError("rewards and terminals must have identical shapes.")
    if rewards.size == 0:
        return rewards.copy()

    if sparse_failure_reward is not None and np.allclose(
        rewards, sparse_failure_reward, atol=1e-6, rtol=0.0
    ):
        if discount < 1.0:
            return np.full_like(
                rewards,
                sparse_failure_reward / (1.0 - discount),
                dtype=np.float32,
            )

    returns = np.empty_like(rewards, dtype=np.float32)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        accumulator = (
            float(rewards[index])
            + discount * accumulator * (1.0 - float(terminals[index]))
        )
        returns[index] = accumulator
    return returns


def _episode_to_arrays(
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    rewards: List[float],
    next_observations: List[np.ndarray],
    terminals: List[bool],
    config: TrainConfig,
) -> NumpyBatch:
    transformed_rewards = (
        np.asarray(rewards, dtype=np.float32) * float(config.reward_scale)
        + float(config.reward_bias)
    )
    terminals_array = np.asarray(terminals, dtype=np.float32)
    mc_returns = _discounted_return_to_go(
        transformed_rewards,
        terminals_array,
        config.discount,
        config.sparse_failure_reward,
    )
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.clip(
            np.asarray(actions, dtype=np.float32),
            -config.clip_action,
            config.clip_action,
        ),
        "rewards": transformed_rewards,
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "masks": 1.0 - terminals_array,
        "dones": terminals_array.astype(bool),
        "mc_returns": mc_returns,
    }


def _concatenate_batches(batches: Sequence[NumpyBatch]) -> NumpyBatch:
    nonempty = [batch for batch in batches if len(batch["rewards"]) > 0]
    if not nonempty:
        raise ValueError("No transitions were produced by the dataset loader.")
    keys = nonempty[0].keys()
    return {
        key: np.concatenate([batch[key] for batch in nonempty], axis=0)
        for key in keys
    }


def _get_raw_d4rl_dataset(env: gym.Env) -> Mapping[str, np.ndarray]:
    candidate = env
    for _ in range(16):
        if hasattr(candidate, "get_dataset"):
            return candidate.get_dataset()
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    raise AttributeError("Environment does not expose get_dataset().")


def _load_d4rl_trajectory_dataset(
    env: gym.Env,
    config: TrainConfig,
    raw_dataset: Optional[Mapping[str, np.ndarray]] = None,
) -> NumpyBatch:
    """Build the Cal-QL training dataset with the released timeout semantics.

    For AntMaze, ``mc_returns`` is the actual behavior-trajectory return-to-go,
    matching the official loader. For environments using the fitted-SARSA
    reference, these values are temporary and are overwritten by the fitted
    reference Q predictions before Cal-QL training starts.
    """
    if raw_dataset is None:
        raw_dataset = _get_raw_d4rl_dataset(env)

    observations = np.asarray(raw_dataset["observations"])
    actions = np.asarray(raw_dataset["actions"])
    rewards = np.asarray(raw_dataset["rewards"])
    terminals = np.asarray(raw_dataset["terminals"]).astype(bool)
    use_timeouts = "timeouts" in raw_dataset
    timeouts = (
        np.asarray(raw_dataset["timeouts"]).astype(bool)
        if use_timeouts
        else None
    )
    environment_horizon = _max_episode_steps(env, config.max_episode_steps)

    episodes: List[NumpyBatch] = []
    ep_obs: List[np.ndarray] = []
    ep_actions: List[np.ndarray] = []
    ep_rewards: List[float] = []
    ep_next_obs: List[np.ndarray] = []
    ep_terminals: List[bool] = []

    total = len(rewards)
    episode_step = 0
    for index in range(total):
        final_timestep = (
            bool(timeouts[index])
            if use_timeouts and timeouts is not None
            else episode_step == environment_horizon - 1
        )
        terminal = bool(terminals[index])

        # This follows the author-released D4RL loader: a timeout transition and
        # the final raw element are not inserted when terminate_on_end=False.
        if not final_timestep and index != total - 1:
            ep_obs.append(observations[index])
            ep_actions.append(actions[index])
            ep_rewards.append(float(rewards[index]))
            if "next_observations" in raw_dataset:
                ep_next_obs.append(np.asarray(raw_dataset["next_observations"][index]))
            else:
                ep_next_obs.append(observations[index + 1])
            ep_terminals.append(terminal)

        episode_step += 1
        if (terminal or final_timestep) and episode_step > 0:
            if ep_rewards:
                episodes.append(
                    _episode_to_arrays(
                        ep_obs,
                        ep_actions,
                        ep_rewards,
                        ep_next_obs,
                        ep_terminals,
                        config,
                    )
                )
            ep_obs, ep_actions, ep_rewards = [], [], []
            ep_next_obs, ep_terminals = [], []
            episode_step = 0

    return _concatenate_batches(episodes)


def _build_sarsa_reference_dataset(
    env: gym.Env,
    config: TrainConfig,
    raw_dataset: Mapping[str, np.ndarray],
) -> NumpyBatch:
    """Create one-step SARSA tuples from the fixed offline dataset.

    Appendix D of Cal-QL specifies a fitted SARSA Q-function for D4RL
    locomotion. A SARSA target requires the behavior action at the next state.
    We therefore only bootstrap when the next raw action belongs to the same
    trajectory. Terminal transitions are retained with zero bootstrap; timeout
    transitions are skipped, matching D4RL's standard ``terminate_on_end=False``
    treatment and avoiding a spurious bootstrap into the next episode's reset.
    """
    observations = np.asarray(raw_dataset["observations"], dtype=np.float32)
    actions = np.asarray(raw_dataset["actions"], dtype=np.float32)
    rewards = np.asarray(raw_dataset["rewards"], dtype=np.float32)
    terminals = np.asarray(raw_dataset["terminals"]).astype(bool)
    timeouts = (
        np.asarray(raw_dataset["timeouts"]).astype(bool)
        if "timeouts" in raw_dataset
        else None
    )
    if "next_observations" in raw_dataset:
        next_observations_all = np.asarray(
            raw_dataset["next_observations"], dtype=np.float32
        )
    else:
        next_observations_all = None

    horizon = _max_episode_steps(env, config.max_episode_steps)
    obs_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []
    next_obs_list: List[np.ndarray] = []
    next_action_list: List[np.ndarray] = []
    mask_list: List[float] = []

    episode_step = 0
    n = len(rewards)
    for i in range(n):
        terminal = bool(terminals[i])
        timeout = (
            bool(timeouts[i])
            if timeouts is not None
            else episode_step == horizon - 1
        )

        if timeout or i == n - 1:
            episode_step += 1
            if terminal or timeout:
                episode_step = 0
            continue

        next_obs = (
            next_observations_all[i]
            if next_observations_all is not None
            else observations[i + 1]
        )
        transformed_reward = (
            float(rewards[i]) * float(config.reward_scale) + float(config.reward_bias)
        )

        if terminal:
            next_action = np.zeros_like(actions[i], dtype=np.float32)
            bootstrap_mask = 0.0
        else:
            # Verify that row i+1 continues the same trajectory. This guards
            # older D4RL datasets that may not provide an explicit timeout flag.
            contiguous = np.allclose(
                observations[i + 1], next_obs, atol=1e-6, rtol=0.0
            )
            if not contiguous:
                episode_step += 1
                continue
            next_action = actions[i + 1]
            bootstrap_mask = 1.0

        obs_list.append(observations[i])
        action_list.append(np.clip(actions[i], -config.clip_action, config.clip_action))
        reward_list.append(transformed_reward)
        next_obs_list.append(next_obs)
        next_action_list.append(
            np.clip(next_action, -config.clip_action, config.clip_action)
        )
        mask_list.append(bootstrap_mask)

        episode_step += 1
        if terminal or timeout:
            episode_step = 0

    if not reward_list:
        raise ValueError("No valid SARSA transitions could be constructed.")

    return {
        "observations": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.float32),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "next_observations": np.asarray(next_obs_list, dtype=np.float32),
        "next_actions": np.asarray(next_action_list, dtype=np.float32),
        "masks": np.asarray(mask_list, dtype=np.float32),
    }


class ArrayDataset:
    def __init__(self, arrays: Mapping[str, np.ndarray], seed: Optional[int] = None):
        if not arrays:
            raise ValueError("Dataset cannot be empty.")
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("Dataset fields have inconsistent lengths.")
        self.arrays = {key: np.asarray(value) for key, value in arrays.items()}
        self.size = int(lengths.pop())
        self._np_random, self.seed_value = gym.utils.seeding.np_random(seed)

    def sample(self, batch_size: int) -> NumpyBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if hasattr(self._np_random, "integers"):
            indices = self._np_random.integers(self.size, size=batch_size)
        else:
            indices = self._np_random.randint(self.size, size=batch_size)
        return {key: value[indices] for key, value in self.arrays.items()}


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
        self.capacity = int(capacity)
        self.size = 0
        self.insert_index = 0
        self.observations = np.empty(
            (capacity, *observation_space.shape), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, *action_space.shape), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=bool)
        self.mc_returns = np.empty((capacity,), dtype=np.float32)
        self._np_random, self.seed_value = gym.utils.seeding.np_random(seed)

    def insert_batch(self, batch: NumpyBatch) -> None:
        length = len(batch["rewards"])
        for index in range(length):
            slot = self.insert_index
            self.observations[slot] = batch["observations"][index]
            self.actions[slot] = batch["actions"][index]
            self.rewards[slot] = batch["rewards"][index]
            self.masks[slot] = batch["masks"][index]
            self.dones[slot] = batch["dones"][index]
            self.next_observations[slot] = batch["next_observations"][index]
            self.mc_returns[slot] = batch["mc_returns"][index]
            self.insert_index = (slot + 1) % self.capacity
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
            "mc_returns": self.mc_returns[indices],
        }


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------


def _init_linear(layer: nn.Linear, gain: float, orthogonal: bool) -> None:
    if orthogonal:
        nn.init.orthogonal_(layer.weight, gain=gain)
    else:
        nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        orthogonal_init: bool,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            _init_linear(linear, math.sqrt(2.0), orthogonal_init)
            layers.extend([linear, nn.ReLU()])
            current_dim = hidden_dim
        output = nn.Linear(current_dim, output_dim)
        _init_linear(output, 1e-2, orthogonal_init)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
        log_std_multiplier_init: float,
        log_std_offset_init: float,
        orthogonal_init: bool,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = MLP(
            observation_dim,
            hidden_dims,
            2 * action_dim,
            orthogonal_init,
        )
        # The official Flax Scalar modules are trainable parameters.
        self.log_std_multiplier = nn.Parameter(
            torch.tensor(float(log_std_multiplier_init), dtype=torch.float32)
        )
        self.log_std_offset = nn.Parameter(
            torch.tensor(float(log_std_offset_init), dtype=torch.float32)
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def distribution(self, observations: torch.Tensor) -> Normal:
        output = self.trunk(observations)
        mean, raw_log_std = torch.chunk(output, 2, dim=-1)
        log_std = (
            raw_log_std * self.log_std_multiplier + self.log_std_offset
        ).clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _log_prob_from_pre_tanh(
        distribution: Normal, pre_tanh: torch.Tensor
    ) -> torch.Tensor:
        base_log_prob = distribution.log_prob(pre_tanh)
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (base_log_prob - correction).sum(dim=-1)

    def sample(
        self,
        observations: torch.Tensor,
        repeat: Optional[int] = None,
        reparameterize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = observations.unsqueeze(1).expand(
                *observations.shape[:-1], repeat, observations.shape[-1]
            )
        distribution = self.distribution(observations)
        pre_tanh = (
            distribution.rsample() if reparameterize else distribution.sample()
        )
        actions = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(distribution, pre_tanh)
        return actions, log_prob

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observations).mean)

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
        return self.mode(tensor).squeeze(0).cpu().numpy()


class QFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        orthogonal_init: bool,
    ):
        super().__init__()
        self.network = MLP(
            observation_dim + action_dim,
            hidden_dims,
            1,
            orthogonal_init,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == observations.ndim + 1:
            observations = observations.unsqueeze(-2).expand(
                *actions.shape[:-1], observations.shape[-1]
            )
        values = self.network(torch.cat([observations, actions], dim=-1))
        return values.squeeze(-1)




def _fit_sarsa_reference(
    dataset: ArrayDataset,
    observation_dim: int,
    action_dim: int,
    config: TrainConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
) -> QFunction:
    """Fit the behavior-policy reference Q by one-step SARSA regression.

    Cal-QL Appendix D explicitly uses a fitted SARSA Q-function for D4RL
    locomotion. The paper does not publish optimizer/training-count details for
    this auxiliary fit, so those quantities are exposed in TrainConfig instead
    of being hidden environment-specific constants.
    """
    q_ref = QFunction(
        observation_dim,
        action_dim,
        config.reference_hidden_dims,
        config.orthogonal_init,
    ).to(device)
    q_target = deepcopy(q_ref).to(device)
    q_target.requires_grad_(False)
    optimizer = torch.optim.Adam(
        q_ref.parameters(), lr=config.reference_learning_rate
    )

    progress = trange(
        config.reference_train_steps,
        desc="Fit SARSA reference Q",
    )
    last_loss = float("nan")
    for step in progress:
        batch_np = dataset.sample(config.reference_batch_size)
        obs = torch.as_tensor(
            batch_np["observations"], dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            batch_np["actions"], dtype=torch.float32, device=device
        )
        rewards = torch.as_tensor(
            batch_np["rewards"], dtype=torch.float32, device=device
        ).reshape(-1)
        next_obs = torch.as_tensor(
            batch_np["next_observations"], dtype=torch.float32, device=device
        )
        next_actions = torch.as_tensor(
            batch_np["next_actions"], dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(
            batch_np["masks"], dtype=torch.float32, device=device
        ).reshape(-1)

        with torch.no_grad():
            target = rewards + config.discount * masks * q_target(
                next_obs, next_actions
            )
        prediction = q_ref(obs, actions)
        loss = F.mse_loss(prediction, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % config.reference_target_update_period == 0:
            _soft_update(q_target, q_ref, config.reference_tau)

        last_loss = float(loss.detach().item())
        if writer is not None and step % config.log_every == 0:
            writer.add_scalar("reference-sarsa/loss", last_loss, step)
            writer.add_scalar(
                "reference-sarsa/q_mean",
                float(prediction.detach().mean().item()),
                step,
            )
            writer.add_scalar(
                "reference-sarsa/target_mean",
                float(target.detach().mean().item()),
                step,
            )

    q_ref.eval()
    q_ref.requires_grad_(False)
    print(
        f"[reference] mode=sarsa steps={config.reference_train_steps} "
        f"final_loss={last_loss:.6f}",
        flush=True,
    )
    return q_ref


@torch.no_grad()
def _predict_reference_values(
    q_ref: QFunction,
    observations: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    chunk_size: int = 16384,
) -> np.ndarray:
    values: List[np.ndarray] = []
    for start in range(0, len(observations), chunk_size):
        end = min(start + chunk_size, len(observations))
        obs = torch.as_tensor(
            observations[start:end], dtype=torch.float32, device=device
        )
        act = torch.as_tensor(
            actions[start:end], dtype=torch.float32, device=device
        )
        values.append(q_ref(obs, act).cpu().numpy().astype(np.float32))
    return np.concatenate(values, axis=0)


# -----------------------------------------------------------------------------
# Cal-QL learner
# -----------------------------------------------------------------------------


@contextmanager
def _frozen_parameters(modules: Iterable[nn.Module]):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


class CalQLLearner:
    def __init__(
        self,
        actor: TanhGaussianPolicy,
        qf1: QFunction,
        qf2: QFunction,
        config: TrainConfig,
        action_dim: int,
        device: torch.device,
    ):
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = deepcopy(qf1).to(device)
        self.target_qf2 = deepcopy(qf2).to(device)
        self.target_qf1.requires_grad_(False)
        self.target_qf2.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.policy_learning_rate
        )
        self.qf1_optimizer = torch.optim.Adam(
            self.qf1.parameters(), lr=config.critic_learning_rate
        )
        self.qf2_optimizer = torch.optim.Adam(
            self.qf2.parameters(), lr=config.critic_learning_rate
        )

        self.log_alpha = torch.tensor(
            math.log(config.init_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.temperature_learning_rate
        )

        self.log_alpha_prime: Optional[torch.Tensor]
        self.alpha_prime_optimizer: Optional[torch.optim.Optimizer]
        if bool(config.cql_lagrange):
            self.log_alpha_prime = torch.tensor(
                config.cql_alpha_prime_init_log,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_prime_optimizer = torch.optim.Adam(
                [self.log_alpha_prime],
                lr=(
                    config.critic_learning_rate
                    if config.cql_lagrange_learning_rate is None
                    else config.cql_lagrange_learning_rate
                ),
            )
        else:
            self.log_alpha_prime = None
            self.alpha_prime_optimizer = None

        self.target_entropy = (
            -float(action_dim)
            if config.target_entropy is None
            else float(config.target_entropy)
        )
        self.config = config
        self.action_dim = int(action_dim)
        self.device = device

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def alpha_prime(self) -> Optional[torch.Tensor]:
        if self.log_alpha_prime is None:
            return None
        return self.log_alpha_prime.exp().clamp(
            min=0.0, max=self.config.cql_alpha_prime_max
        )

    def _target_q(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            if self.config.max_target_backup:
                next_actions, next_log_prob = self.actor.sample(
                    batch["next_observations"],
                    repeat=self.config.target_backup_n_actions,
                    reparameterize=False,
                )
                target_q1 = self.target_qf1(
                    batch["next_observations"], next_actions
                )
                target_q2 = self.target_qf2(
                    batch["next_observations"], next_actions
                )
                target_q_candidates = torch.minimum(target_q1, target_q2)
                # Official code selects the action with maximal target Q first,
                # then gathers that action's log-probability for optional entropy
                # backup. It does not maximize Q - alpha log pi directly.
                max_indices = target_q_candidates.argmax(dim=-1, keepdim=True)
                target_q = target_q_candidates.gather(-1, max_indices).squeeze(-1)
                selected_next_log_prob = next_log_prob.gather(
                    -1, max_indices
                ).squeeze(-1)
                if self.config.backup_entropy:
                    target_q = (
                        target_q
                        - self.alpha.detach() * selected_next_log_prob
                    )
            else:
                next_actions, next_log_prob = self.actor.sample(
                    batch["next_observations"], reparameterize=False
                )
                target_q = torch.minimum(
                    self.target_qf1(batch["next_observations"], next_actions),
                    self.target_qf2(batch["next_observations"], next_actions),
                )
                if self.config.backup_entropy:
                    target_q = target_q - self.alpha.detach() * next_log_prob

            return (
                batch["rewards"]
                + self.config.discount * batch["masks"] * target_q
            )

    def _cql_losses(
        self,
        batch: TorchBatch,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = batch["observations"].shape[0]
        n_actions = self.config.cql_n_actions

        random_actions = torch.empty(
            batch_size,
            n_actions,
            self.action_dim,
            device=self.device,
        ).uniform_(-1.0, 1.0)
        random_log_density = self.action_dim * math.log(0.5)

        with torch.no_grad():
            current_actions, current_log_probs = self.actor.sample(
                batch["observations"], repeat=n_actions, reparameterize=False
            )
            next_actions, next_log_probs = self.actor.sample(
                batch["next_observations"], repeat=n_actions, reparameterize=False
            )

        q1_random = self.qf1(batch["observations"], random_actions)
        q2_random = self.qf2(batch["observations"], random_actions)
        q1_current = self.qf1(batch["observations"], current_actions)
        q2_current = self.qf2(batch["observations"], current_actions)
        # The official CQL proposal evaluates actions sampled at s' on s.
        q1_next = self.qf1(batch["observations"], next_actions)
        q2_next = self.qf2(batch["observations"], next_actions)

        if self.config.enable_calql:
            lower_bound = batch["mc_returns"].unsqueeze(-1)
            q1_current = torch.maximum(q1_current, lower_bound)
            q2_current = torch.maximum(q2_current, lower_bound)
            q1_next = torch.maximum(q1_next, lower_bound)
            q2_next = torch.maximum(q2_next, lower_bound)

        if self.config.cql_importance_sample:
            q1_candidates = torch.cat(
                [
                    q1_random - random_log_density,
                    q1_next - next_log_probs,
                    q1_current - current_log_probs,
                ],
                dim=-1,
            )
            q2_candidates = torch.cat(
                [
                    q2_random - random_log_density,
                    q2_next - next_log_probs,
                    q2_current - current_log_probs,
                ],
                dim=-1,
            )
        else:
            q1_candidates = torch.cat(
                [q1_random, q1_data.unsqueeze(-1), q1_next, q1_current], dim=-1
            )
            q2_candidates = torch.cat(
                [q2_random, q2_data.unsqueeze(-1), q2_next, q2_current], dim=-1
            )

        temperature = self.config.cql_temperature
        q1_ood = temperature * torch.logsumexp(
            q1_candidates / temperature, dim=-1
        )
        q2_ood = temperature * torch.logsumexp(
            q2_candidates / temperature, dim=-1
        )
        q1_diff = (q1_ood - q1_data).clamp(
            self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
        )
        q2_diff = (q2_ood - q2_data).clamp(
            self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
        )

        if bool(self.config.cql_lagrange):
            alpha_prime = self.alpha_prime
            if alpha_prime is None:
                raise RuntimeError("Lagrange CQL is enabled without alpha-prime.")
            gap = float(self.config.cql_target_action_gap)
            q1_cql_loss = (
                alpha_prime.detach()
                * float(self.config.cql_min_q_weight)
                * (q1_diff.mean() - gap)
            )
            q2_cql_loss = (
                alpha_prime.detach()
                * float(self.config.cql_min_q_weight)
                * (q2_diff.mean() - gap)
            )
            alpha_prime_loss = -0.5 * alpha_prime * float(
                self.config.cql_min_q_weight
            ) * (
                (q1_diff.detach().mean() - gap)
                + (q2_diff.detach().mean() - gap)
            )
        else:
            q1_cql_loss = float(self.config.cql_min_q_weight) * q1_diff.mean()
            q2_cql_loss = float(self.config.cql_min_q_weight) * q2_diff.mean()
            alpha_prime_loss = torch.zeros((), device=self.device)

        info = {
            "cql_q1_diff": q1_diff.mean(),
            "cql_q2_diff": q2_diff.mean(),
            "cql_q1_ood": q1_ood.mean(),
            "cql_q2_ood": q2_ood.mean(),
            "cql_alpha_prime_loss": alpha_prime_loss,
        }
        return q1_cql_loss, q2_cql_loss, info

    def update(self, batch: TorchBatch, use_cql: bool = True) -> Dict[str, float]:
        # Build all losses before stepping optimizers so actor/Q/temperature are
        # evaluated from the same pre-update parameter state, matching JAX's
        # functional update semantics closely.
        sampled_actions, log_pi = self.actor.sample(
            batch["observations"], reparameterize=True
        )
        with _frozen_parameters((self.qf1, self.qf2)):
            policy_q = torch.minimum(
                self.qf1(batch["observations"], sampled_actions),
                self.qf2(batch["observations"], sampled_actions),
            )
        actor_loss = (self.alpha.detach() * log_pi - policy_q).mean()
        alpha_loss = -(
            self.log_alpha * (log_pi.detach() + self.target_entropy)
        ).mean()

        target_q = self._target_q(batch)
        q1_data = self.qf1(batch["observations"], batch["actions"])
        q2_data = self.qf2(batch["observations"], batch["actions"])
        qf1_bellman_loss = F.mse_loss(q1_data, target_q)
        qf2_bellman_loss = F.mse_loss(q2_data, target_q)

        cql_info: Dict[str, torch.Tensor] = {}
        if use_cql:
            qf1_cql_loss, qf2_cql_loss, cql_info = self._cql_losses(
                batch, q1_data, q2_data
            )
        else:
            qf1_cql_loss = torch.zeros((), device=self.device)
            qf2_cql_loss = torch.zeros((), device=self.device)
            cql_info["cql_alpha_prime_loss"] = torch.zeros((), device=self.device)

        qf1_loss = qf1_bellman_loss + qf1_cql_loss
        qf2_loss = qf2_bellman_loss + qf2_cql_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        self.qf1_optimizer.zero_grad(set_to_none=True)
        self.qf2_optimizer.zero_grad(set_to_none=True)
        if self.alpha_prime_optimizer is not None:
            self.alpha_prime_optimizer.zero_grad(set_to_none=True)

        actor_loss.backward()
        alpha_loss.backward()
        qf1_loss.backward()
        qf2_loss.backward()
        if use_cql and self.alpha_prime_optimizer is not None:
            cql_info["cql_alpha_prime_loss"].backward()

        self.actor_optimizer.step()
        self.alpha_optimizer.step()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        if use_cql and self.alpha_prime_optimizer is not None:
            self.alpha_prime_optimizer.step()

        _soft_update(self.target_qf1, self.qf1, self.config.tau)
        _soft_update(self.target_qf2, self.qf2, self.config.tau)

        metrics = {
            "actor_loss": float(actor_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "qf1_loss": float(qf1_loss.detach().item()),
            "qf2_loss": float(qf2_loss.detach().item()),
            "qf1_bellman_loss": float(qf1_bellman_loss.detach().item()),
            "qf2_bellman_loss": float(qf2_bellman_loss.detach().item()),
            "qf1_cql_loss": float(qf1_cql_loss.detach().item()),
            "qf2_cql_loss": float(qf2_cql_loss.detach().item()),
            "q_data": float(torch.minimum(q1_data, q2_data).detach().mean().item()),
            "target_q": float(target_q.detach().mean().item()),
            "entropy": float((-log_pi).detach().mean().item()),
        }
        for key, value in cql_info.items():
            metrics[key] = float(value.detach().item())
        if self.alpha_prime is not None:
            metrics["cql_alpha_prime"] = float(self.alpha_prime.detach().item())
        return metrics


# -----------------------------------------------------------------------------
# Batch handling, trajectories, and evaluation
# -----------------------------------------------------------------------------


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
        "mc_returns": torch.as_tensor(
            batch["mc_returns"], dtype=torch.float32, device=device
        ).reshape(-1),
    }


def _mixed_batch(
    offline_dataset: ArrayDataset,
    online_buffer: ReplayBuffer,
    batch_size: int,
    offline_ratio: float,
) -> NumpyBatch:
    if online_buffer.size == 0 or offline_ratio >= 1.0:
        return offline_dataset.sample(batch_size)
    offline_size = int(batch_size * offline_ratio)
    offline_size = min(max(offline_size, 0), batch_size)
    online_size = batch_size - offline_size

    parts = []
    if offline_size > 0:
        parts.append(offline_dataset.sample(offline_size))
    if online_size > 0:
        parts.append(online_buffer.sample(online_size))
    if len(parts) == 1:
        return parts[0]
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }


def _is_goal_achieved(info: Mapping) -> bool:
    value = info.get("goal_achieved", False)
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def collect_trajectory(
    actor: TanhGaussianPolicy,
    env: gym.Env,
    device: torch.device,
    config: TrainConfig,
    interaction_limit: Optional[int] = None,
) -> Tuple[NumpyBatch, float, int]:
    observation = _reset_env(env)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    next_observations: List[np.ndarray] = []
    terminals: List[bool] = []
    raw_return = 0.0
    maximum_steps = _max_episode_steps(env, config.max_episode_steps)
    if interaction_limit is not None:
        maximum_steps = min(maximum_steps, int(interaction_limit))

    for step in range(maximum_steps):
        action = actor.sample_action(observation, device)
        action = np.clip(action, -config.clip_action, config.clip_action)
        next_observation, reward, done, info = _step_env(env, action)
        timeout = bool(info.get("TimeLimit.truncated", False))
        terminal = bool(done and not timeout)

        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_observation, dtype=np.float32))
        terminals.append(terminal)
        raw_return += reward
        observation = next_observation

        if done:
            break

    batch = _episode_to_arrays(
        observations,
        actions,
        rewards,
        next_observations,
        terminals,
        config,
    )
    return batch, float(raw_return), len(rewards)


@torch.no_grad()
def evaluate(
    actor: TanhGaussianPolicy,
    env: gym.Env,
    device: torch.device,
    num_episodes: int,
    max_episode_steps: Optional[int],
) -> Dict[str, float]:
    actor.eval()
    returns = []
    lengths = []
    for _ in range(num_episodes):
        observation = _reset_env(env)
        episode_return = 0.0
        episode_length = 0
        done = False
        maximum_steps = _max_episode_steps(env, max_episode_steps)
        while not done and episode_length < maximum_steps:
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


def _build_offline_dataset(
    env: gym.Env,
    config: TrainConfig,
) -> Tuple[ArrayDataset, Optional[ArrayDataset]]:
    raw_dataset = _get_raw_d4rl_dataset(env)
    arrays = _load_d4rl_trajectory_dataset(env, config, raw_dataset)
    dataset_seed = config.train_seed if config.dataset_seed is None else config.dataset_seed
    offline_dataset = ArrayDataset(arrays, seed=dataset_seed)

    sarsa_dataset: Optional[ArrayDataset] = None
    if config.reference_mode == "sarsa":
        sarsa_arrays = _build_sarsa_reference_dataset(env, config, raw_dataset)
        sarsa_dataset = ArrayDataset(sarsa_arrays, seed=dataset_seed + 17)
    return offline_dataset, sarsa_dataset


def _write_metrics(
    writer: SummaryWriter,
    prefix: str,
    metrics: Mapping[str, float],
    step: int,
) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


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
    observation_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(np.prod(train_env.action_space.shape))

    offline_dataset, sarsa_dataset = _build_offline_dataset(train_env, config)
    online_buffer = ReplayBuffer(
        train_env.observation_space,
        train_env.action_space,
        config.replay_buffer_size,
        seed=config.train_seed,
    )

    run_name = f"{config.env_name}_seed{config.train_seed}_CalQL"
    log_dir = os.path.join(config.log_root, config.env_name, f"seed_{config.train_seed}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Appendix D: for non-terminal D4RL datasets, replace the temporary MC field
    # by predictions from a fitted SARSA Q-function before Cal-QL starts.
    reference_q: Optional[QFunction] = None
    if config.reference_mode == "sarsa":
        if sarsa_dataset is None:
            raise RuntimeError("reference_mode=sarsa but no SARSA dataset was built.")
        reference_q = _fit_sarsa_reference(
            sarsa_dataset,
            observation_dim,
            action_dim,
            config,
            device,
            writer,
        )
        offline_dataset.arrays["mc_returns"] = _predict_reference_values(
            reference_q,
            offline_dataset.arrays["observations"],
            offline_dataset.arrays["actions"],
            device,
        )
        writer.add_scalar(
            "reference-sarsa/offline_reference_mean",
            float(np.mean(offline_dataset.arrays["mc_returns"])),
            config.reference_train_steps,
        )
        writer.add_scalar(
            "reference-sarsa/offline_reference_std",
            float(np.std(offline_dataset.arrays["mc_returns"])),
            config.reference_train_steps,
        )
        writer.flush()

    actor = TanhGaussianPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=config.actor_hidden_dims,
        log_std_min=config.actor_log_std_min,
        log_std_max=config.actor_log_std_max,
        log_std_multiplier_init=config.actor_log_std_multiplier_init,
        log_std_offset_init=config.actor_log_std_offset_init,
        orthogonal_init=config.orthogonal_init,
    ).to(device)
    qf1 = QFunction(
        observation_dim,
        action_dim,
        config.critic_hidden_dims,
        config.orthogonal_init,
    ).to(device)
    qf2 = QFunction(
        observation_dim,
        action_dim,
        config.critic_hidden_dims,
        config.orthogonal_init,
    ).to(device)
    learner = CalQLLearner(actor, qf1, qf2, config, action_dim, device)

    print(
        f"env={config.env_name} seed={config.train_seed} device={device} "
        f"reference={config.reference_mode} offline_size={offline_dataset.size} "
        f"actor={config.actor_hidden_dims} critic={config.critic_hidden_dims} "
        f"cql_alpha={config.cql_min_q_weight} lagrange={config.cql_lagrange} "
        f"pretrain={config.pretrain_steps} online={config.num_total_steps} "
        f"mix={config.mixing_ratio} utd={config.online_utd_ratio} log={log_dir}",
        flush=True,
    )

    # Initial evaluation before any Cal-QL update.
    eval_info = evaluate(
        actor, eval_env, device, config.eval_episodes, config.max_episode_steps
    )
    _write_metrics(writer, "offline-evaluation", eval_info, 0)
    normalized = _get_normalized_score(eval_env, eval_info["return"])
    if normalized is not None:
        writer.add_scalar("offline-evaluation/d4rl_normalized_score", normalized, 0)
    writer.flush()

    # Offline Cal-QL pre-training.
    for gradient_step in trange(
        1, config.pretrain_steps + 1, desc="Cal-QL offline pre-training"
    ):
        batch = _to_torch_batch(offline_dataset.sample(config.batch_size), device)
        metrics = learner.update(batch, use_cql=True)

        if gradient_step % config.log_every == 0:
            _write_metrics(writer, "offline-training", metrics, gradient_step)
        if gradient_step % config.eval_every == 0 or gradient_step == config.pretrain_steps:
            eval_info = evaluate(
                actor,
                eval_env,
                device,
                config.eval_episodes,
                config.max_episode_steps,
            )
            _write_metrics(writer, "offline-evaluation", eval_info, gradient_step)
            normalized = _get_normalized_score(eval_env, eval_info["return"])
            message = (
                f"[offline] step={gradient_step} return={eval_info['return']:.3f} "
                f"length={eval_info['length']:.1f}"
            )
            if normalized is not None:
                writer.add_scalar(
                    "offline-evaluation/d4rl_normalized_score",
                    normalized,
                    gradient_step,
                )
                message += f" normalized={normalized:.3f}"
            print(message, flush=True)
            writer.flush()

    # Online fine-tuning. As in the released Cal-QL code, complete trajectories
    # are collected before updates. Online trajectories use their own return-to-
    # go as a valid empirical reference; the fixed offline samples retain either
    # MC references (AntMaze) or fitted-SARSA references (locomotion/Adroit).
    environment_steps = 0
    online_gradient_steps = 0
    next_eval_step = config.eval_every
    online_use_cql = bool(config.online_use_cql)
    if config.cql_min_q_weight_online is not None:
        config.cql_min_q_weight = float(config.cql_min_q_weight_online)

    progress = trange(config.num_total_steps, desc="Cal-QL online fine-tuning")
    while environment_steps < config.num_total_steps:
        trajectories_this_iteration = 0
        collected_this_iteration = 0

        while (
            trajectories_this_iteration < config.trajectories_per_iteration
            and environment_steps < config.num_total_steps
        ):
            trajectory, raw_return, trajectory_length = collect_trajectory(
                actor,
                train_env,
                device,
                config,
                interaction_limit=config.num_total_steps - environment_steps,
            )
            online_buffer.insert_batch(trajectory)
            previous_steps = environment_steps
            environment_steps += trajectory_length
            collected_this_iteration += trajectory_length
            trajectories_this_iteration += 1
            progress.update(
                min(
                    trajectory_length,
                    max(config.num_total_steps - previous_steps, 0),
                )
            )
            writer.add_scalar("online-collection/return", raw_return, environment_steps)
            writer.add_scalar(
                "online-collection/length", trajectory_length, environment_steps
            )

        update_count = collected_this_iteration * config.online_utd_ratio
        for _ in range(update_count):
            numpy_batch = _mixed_batch(
                offline_dataset,
                online_buffer,
                config.batch_size,
                config.mixing_ratio,
            )
            metrics = learner.update(
                _to_torch_batch(numpy_batch, device),
                use_cql=online_use_cql,
            )
            online_gradient_steps += 1
            if online_gradient_steps % config.log_every == 0:
                _write_metrics(
                    writer,
                    "online-training",
                    metrics,
                    online_gradient_steps,
                )

        if environment_steps >= next_eval_step or environment_steps >= config.num_total_steps:
            eval_info = evaluate(
                actor,
                eval_env,
                device,
                config.eval_episodes,
                config.max_episode_steps,
            )
            _write_metrics(writer, "online-evaluation", eval_info, environment_steps)
            normalized = _get_normalized_score(eval_env, eval_info["return"])
            message = (
                f"[online] env_step={environment_steps} grad_step={online_gradient_steps} "
                f"return={eval_info['return']:.3f} length={eval_info['length']:.1f}"
            )
            if normalized is not None:
                writer.add_scalar(
                    "online-evaluation/d4rl_normalized_score",
                    normalized,
                    environment_steps,
                )
                message += f" normalized={normalized:.3f}"
            print(message, flush=True)
            writer.flush()
            while next_eval_step <= environment_steps:
                next_eval_step += config.eval_every

    progress.close()

    if config.checkpoints_path is not None:
        checkpoint_dir = os.path.join(config.checkpoints_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "target_qf1": learner.target_qf1.state_dict(),
            "target_qf2": learner.target_qf2.state_dict(),
            "log_alpha": learner.log_alpha.detach().cpu(),
            "log_alpha_prime": (
                None
                if learner.log_alpha_prime is None
                else learner.log_alpha_prime.detach().cpu()
            ),
            "config": asdict(config),
            "environment_steps": environment_steps,
            "online_gradient_steps": online_gradient_steps,
        }
        if reference_q is not None:
            checkpoint["reference_q"] = reference_q.state_dict()
        torch.save(checkpoint, os.path.join(checkpoint_dir, "final.pt"))

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train(pyrallis.parse(config_class=TrainConfig))
