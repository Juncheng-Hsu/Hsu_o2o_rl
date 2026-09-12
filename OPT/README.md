# OPT PyTorch

`opt.py` 是用于 Offline-to-Online Reinforcement Learning（O2O RL）实验的单文件 PyTorch 实现。

算法实现依据：

- Shin et al., **Online Pre-Training for Offline-to-Online Reinforcement Learning**, ICML 2025.
- 作者官方 PyTorch 实现：`LGAI-Research/opt`.

本文件的算法逻辑以 OPT 论文和作者官方代码为准；项目中原有 OPT 代码只作为代码组织和日志风格参考。

当前实现直接支持与本项目 RLPD 实验对应的 30 个 D4RL 环境，并根据 `env_name` 自动设置环境相关的 OPT / TD3+BC / SPOT 参数。

---

## 1. 基本运行方式

最简单的运行方式：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0
```

指定 GPU：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --device="cuda:0"
```

AntMaze：

```bash
python opt.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

Adroit：

```bash
python opt.py --env_name="door-cloned-v1" --train_seed=0 --device="cuda:0"
```

正式 baseline 实验建议保持：

```text
auto_env_config = True
```

这样代码会根据 `env_name` 自动加载环境对应的 OPT 参数。

---

## 2. 支持的 30 个环境

### MuJoCo Locomotion

```text
halfcheetah-random-v2
halfcheetah-medium-v2
halfcheetah-medium-replay-v2
halfcheetah-medium-expert-v2

hopper-random-v2
hopper-medium-v2
hopper-medium-replay-v2
hopper-medium-expert-v2

walker2d-random-v2
walker2d-medium-v2
walker2d-medium-replay-v2
walker2d-medium-expert-v2
```

### AntMaze

```text
antmaze-umaze-v2
antmaze-umaze-diverse-v2
antmaze-medium-play-v2
antmaze-medium-diverse-v2
antmaze-large-play-v2
antmaze-large-diverse-v2
```

### D4RL Adroit

```text
pen-human-v1
pen-cloned-v1
pen-expert-v1

door-human-v1
door-cloned-v1
door-expert-v1

hammer-human-v1
hammer-cloned-v1
hammer-expert-v1

relocate-human-v1
relocate-cloned-v1
relocate-expert-v1
```

---

## 3. 官方环境与适配环境

OPT 原论文/官方配置并没有覆盖上述全部 30 个环境。

代码会在启动时打印：

```text
config_source=official
```

或者：

```text
config_source=adapted
```

### 3.1 官方覆盖环境

共 19 个。

#### MuJoCo

```text
halfcheetah-random-v2
halfcheetah-medium-v2
halfcheetah-medium-replay-v2

hopper-random-v2
hopper-medium-v2
hopper-medium-replay-v2

walker2d-random-v2
walker2d-medium-v2
walker2d-medium-replay-v2
```

#### AntMaze

```text
antmaze-umaze-v2
antmaze-umaze-diverse-v2
antmaze-medium-play-v2
antmaze-medium-diverse-v2
antmaze-large-play-v2
antmaze-large-diverse-v2
```

#### Adroit

```text
pen-cloned-v1
door-cloned-v1
hammer-cloned-v1
relocate-cloned-v1
```

这些环境启动时显示：

```text
config_source=official
```

---

### 3.2 为匹配 RLPD benchmark 而增加的适配环境

共 11 个。

#### MuJoCo medium-expert

```text
halfcheetah-medium-expert-v2
hopper-medium-expert-v2
walker2d-medium-expert-v2
```

固定迁移规则：

```text
使用同域 medium 的 OPT 参数
```

即：

```text
TD3+BC + OPT
kappa = 0.3 -> 0.9
kappa_cool_steps = 150000
UTD = 5
```

不针对单个环境额外调参。

#### Adroit human / expert

```text
pen-human-v1
pen-expert-v1
door-human-v1
door-expert-v1
hammer-human-v1
hammer-expert-v1
relocate-human-v1
relocate-expert-v1
```

固定迁移规则：

```text
使用官方 Adroit-cloned 的 SPOT+OPT 域级配置
```

不针对 human / expert 单独调参。

这些环境启动时显示：

```text
config_source=adapted
```

并额外打印提示，说明该任务没有出现在 OPT 原论文的正式 benchmark 中。

---

## 4. 严格只运行官方环境

如果只想允许原论文/官方配置覆盖的环境，可以运行：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --strict_official_only=True
```

如果环境属于适配任务，例如：

```bash
python opt.py --env_name="hopper-medium-expert-v2" --train_seed=0 --strict_official_only=True
```

程序会直接报错，而不是静默使用迁移参数。

正式复现 OPT 原论文时，可以打开：

```text
strict_official_only = True
```

本项目为了与 RLPD 的 30 个任务保持一致时，应使用默认：

```text
strict_official_only = False
```

---

## 5. OPT 核心训练流程

当前实现保留 OPT 的主要算法结构。

整体流程：

```text
Offline dataset
      ↓
Offline RL pre-training
      ↓
得到 π_off 和 Q_off
      ↓
使用冻结的 offline policy 收集 N_tau 个 online transitions
      ↓
初始化独立的 Q_on
      ↓
OPT Online Pre-Training
      ↓
同时维护 Q_off 与 Q_on
      ↓
Density-ratio balanced replay
      ↓
Actor 使用混合价值进行更新
      ↓
Online fine-tuning
```

Actor 在线目标中的核心价值为：

```text
Q_mix = (1 - kappa) * Q_off + kappa * Q_on
```

即：

```math
Q_{\text{mix}}(s,a)
=
(1-\kappa)Q_{\text{off}}(s,a)
+
\kappa Q_{\text{on}}(s,a).
```

这里：

```text
Q_off
```

表示继承 offline pre-training 的 critic。

```text
Q_on
```

表示重新初始化、经过 OPT online pre-training 的 critic。

---

## 6. Online Pre-Training

所有环境默认：

```text
n_tau = 25000
online_pretrain_steps = 50000
online_pretrain_inner_lr = 3e-4
```

即先使用 offline policy 收集：

```text
25000 online transitions
```

在这段期间不会执行常规 online actor/critic fine-tuning。

达到：

```text
online_buffer.size >= 25000
```

后执行一次：

```text
50000 steps
```

的 OPT online pre-training。

核心过程是：

```text
offline TD loss
      ↓
differentiable inner update
      ↓
online TD loss
      ↓
meta-gradient update
```

代码通过 `HotPlug` 实现作者官方代码中的可微参数 inner update，而不是简单重新训练一个普通 critic。

---

## 7. Density-Ratio Balanced Replay

在线阶段维护：

```text
offline_buffer
online_buffer
priority_buffer
```

其中 `priority_buffer` 初始包含整个 offline dataset，之后继续加入所有 online transitions。

OPT 使用 density-ratio 网络估计 offline / online 数据分布差异，并据此调整 replay priority。

默认：

```text
priority_temperature = 5.0
density_learning_rate = 3e-4
priority_replay_buffer_size = 1000000
```

这一部分是 OPT 的组成部分，不应删除后仍将算法称为完整 OPT。

---

# 8. 环境参数自动配置

正式 baseline 建议保持：

```text
auto_env_config=True
```

不要通过命令行逐环境手工修改 OPT 的环境参数。

实验协议参数和环境算法参数是分开的。

---

## 9. MuJoCo 自动配置

所有 MuJoCo 环境使用：

```text
algo_type = TD3

hidden_dim = 256

actor_learning_rate = 3e-4
critic_learning_rate = 3e-4

actor_layer_norm = False
critic_layer_norm = False

batch_size = 256

discount = 0.99
tau = 0.005

exploration_noise = 0.1
policy_noise = 0.2
noise_clip = 0.5
policy_frequency = 2

td3_bc_alpha = 2.5

normalize_states = False
normalize_rewards = False

priority_temperature = 5.0

n_tau = 25000
online_pretrain_steps = 50000

utd_ratio = 5
```

### random

例如：

```text
halfcheetah-random-v2
hopper-random-v2
walker2d-random-v2
```

自动设置：

```text
kappa = 1.0
kappa_end = 1.0
kappa_cool_steps = 0
```

即 actor 在线阶段直接主要依赖重新 online-pretrained 的 critic。

---

### medium

例如：

```text
halfcheetah-medium-v2
hopper-medium-v2
walker2d-medium-v2
```

自动设置：

```text
kappa = 0.3
kappa_end = 0.9
kappa_cool_steps = 150000
```

---

### medium-replay

例如：

```text
halfcheetah-medium-replay-v2
hopper-medium-replay-v2
walker2d-medium-replay-v2
```

自动设置：

```text
kappa = 0.1
kappa_end = 0.9
kappa_cool_steps = 150000
```

---

### medium-expert

该组不在 OPT 原论文正式 benchmark 中。

固定采用：

```text
kappa = 0.3
kappa_end = 0.9
kappa_cool_steps = 150000
```

即使用 medium 的 schedule。

启动时：

```text
config_source=adapted
```

---

## 10. AntMaze 自动配置

AntMaze 使用：

```text
algo_type = SPOT

hidden_dim = 256

actor_learning_rate = 1e-4
critic_learning_rate = 3e-4

actor_layer_norm = False
critic_layer_norm = False

actor_init_w = 1e-3
critic_init_w = 3e-3

normalize_rewards = True

vae_learning_rate = 1e-3
vae_hidden_dim = 750
vae_beta = 0.5
vae_steps = 100000

support_lambda_cool = True
support_lambda_end = 0.2

online_discount = 0.995

kappa = 0.1
kappa_end = 0.9

utd_ratio = 1
```

AntMaze reward normalization 按 SPOT/OPT 配置处理：

```text
r_train = r - 1
```

因此 D4RL AntMaze 原始：

```text
failure = 0
success = 1
```

训练时对应：

```text
failure = -1
success = 0
```

---

### Umaze

```text
antmaze-umaze-v2
antmaze-umaze-diverse-v2
```

自动设置：

```text
support_lambda = 0.25
kappa_cool_steps = 100000
```

---

### Medium

```text
antmaze-medium-play-v2
antmaze-medium-diverse-v2
```

自动设置：

```text
support_lambda = 0.05
kappa_cool_steps = 100000
```

---

### Large Play

```text
antmaze-large-play-v2
```

自动设置：

```text
support_lambda = 0.025
kappa_cool_steps = 100000
```

---

### Large Diverse

```text
antmaze-large-diverse-v2
```

自动设置：

```text
support_lambda = 0.025
kappa_cool_steps = 200000
```

---

## 11. Adroit 自动配置

Adroit 使用：

```text
algo_type = SPOT

hidden_dim = 256

actor_learning_rate = 1e-4
critic_learning_rate = 3e-4

actor_layer_norm = True
critic_layer_norm = True

actor_init_w = 1e-3
critic_init_w = 3e-3

normalize_rewards = False

vae_learning_rate = 1e-3
vae_hidden_dim = 750
vae_beta = 0.5
vae_steps = 100000

support_lambda = 1.0
support_lambda_cool = True
support_lambda_end = 0.5

online_discount = 0.99

kappa = 0.1
kappa_end = 0.9
kappa_cool_steps = 250000

utd_ratio = 1
```

其中：

```text
cloned
```

属于官方 OPT benchmark。

```text
human
expert
```

使用固定同域迁移设置，并标记为：

```text
config_source=adapted
```

---

## 12. 实验协议参数

下面这些参数属于你的实验协议，不会被 `env_name` 自动覆盖：

```text
offline_steps
online_steps
eval_every
eval_episodes
log_every

train_seed
eval_seed

device
checkpoints_path
log_root
```

当前默认：

```text
offline_steps = 1000000
online_steps = 1000000

eval_every = 5000
eval_episodes = 10

log_every = 1000

train_seed = 0
eval_seed = 0
```

因此直接运行：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0
```

默认就是：

```text
1M offline gradient updates
+
1M online environment interactions
```

---

## 13. 与统一 1M-step O2O benchmark 对齐

如果你的论文要求所有 baseline：

```text
online steps = 1,000,000
evaluation every 5,000 steps
10 evaluation episodes
```

可以直接：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
```

由于这些正好也是当前默认值，因此以下命令等价：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --device="cuda:0"
```

为了实验记录清晰，正式跑论文结果时仍建议显式写出关键实验协议参数。

---

## 14. 如果复现 OPT 论文的较短 online budget

OPT 原论文的主要结果常在较短的 online interaction budget 下比较。

如果需要单独做论文复现，可手工指定：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --online_steps=300000 --device="cuda:0"
```

不要把这个 300k 结果与本项目统一 1M-step baseline 混在同一张表里而不说明 interaction budget。

---

## 15. SPOT VAE 预训练

AntMaze 和 Adroit 会额外执行：

```text
vae_steps = 100000
```

即：

```text
SPOT VAE pre-training
```

之后才执行：

```text
offline_steps
```

因此对于 SPOT+OPT：

```text
100k VAE updates
+
1M offline policy/critic updates
+
online interaction
```

这里的 VAE 更新不属于 online environment interaction。

---

## 16. UTD

### MuJoCo

```text
utd_ratio = 5
```

达到 `N_tau` 并完成 Online Pre-Training 后，每一个 online environment step 执行：

```text
5 OPT online updates
```

### AntMaze / Adroit

```text
utd_ratio = 1
```

即：

```text
1 online environment step
≈
1 OPT online update
```

注意：前 `N_tau=25000` 个 online interactions 不执行常规 online fine-tuning，它们用于构造 Online Pre-Training 数据，并且包含在统一的 1M online interaction budget 内。常规 online fine-tuning 从下一次交互开始，因此共有 975,000 个可执行常规更新的交互步。

---

## 17. kappa schedule

代码中的：

```text
kappa
```

是：

```text
Q_on
```

在 actor 混合价值中的权重。

即：

```math
Q_{\mathrm{mix}}
=
(1-\kappa)Q_{\mathrm{off}}
+
\kappa Q_{\mathrm{on}}.
```

当：

```text
kappa -> 1
```

时，actor 越来越依赖 online-pretrained critic。

代码按官方 OPT schedule 从初始 `kappa` 逐渐变化到：

```text
kappa_end
```

不要把 `kappa` 理解成 offline critic 的权重。

---

## 18. 推荐正式实验命令

### MuJoCo

```bash
python opt.py --env_name="halfcheetah-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="hopper-medium-replay-v2" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="walker2d-medium-expert-v2" --train_seed=0 --device="cuda:0"
```

### AntMaze

```bash
python opt.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="antmaze-medium-diverse-v2" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="antmaze-large-diverse-v2" --train_seed=0 --device="cuda:0"
```

### Adroit

```bash
python opt.py --env_name="pen-cloned-v1" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="door-human-v1" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="hammer-expert-v1" --train_seed=0 --device="cuda:0"
```

---

## 19. 多随机种子

正式主实验例如使用：

```text
0
10
20
30
40
```

分别运行：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=10 --device="cuda:0"
```

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=20 --device="cuda:0"
```

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=30 --device="cuda:0"
```

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=40 --device="cuda:0"
```

`eval_seed` 与 `train_seed` 是独立参数。

默认：

```text
eval_seed = 0
```

因此不同训练随机种子默认使用相同 evaluation seeds，这有利于减少 evaluation randomness。

如果希望 evaluation seed 随训练 seed 一起变化，需要显式设置：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=10 --eval_seed=10 --device="cuda:0"
```

正式论文中应固定一种规则，不要混用。

---

## 20. 日志目录

默认：

```text
log_root = logs/OPT
```

实际日志目录格式：

```text
logs/OPT/<domain>/<env_name>_seed<seed>_<backbone>_OPT/
```

例如：

```text
logs/OPT/hopper/hopper-medium-v2_seed0_TD3_OPT/
```

AntMaze 示例：

```text
logs/OPT/antmaze/antmaze-umaze-v2_seed0_SPOT_OPT/
```

Adroit 示例：

```text
logs/OPT/door/door-cloned-v1_seed0_SPOT_OPT/
```

启动 TensorBoard：

```bash
tensorboard --logdir logs/OPT --port 8088
```

---

## 21. TensorBoard 主要指标

Offline 阶段：

```text
offline/*
offline_evaluation/return
offline_evaluation/length
offline_evaluation/success
offline_evaluation/normalized_score
```

Online Pre-Training：

```text
online_pretrain/offline_td_loss
online_pretrain/online_td_loss
online_pretrain/gradient_weight
```

Online fine-tuning：

```text
online/offline_critic_loss
online/new_critic_loss
online/density_loss
online/offline_density
online/online_density
online/priority_mean
online/kappa
online/actor_loss
online/old_q
online/new_q
online/blended_q
```

正式论文主要关注：

```text
evaluation/normalized_score
```

以及必要时：

```text
evaluation/return
evaluation/success
```

---

## 22. Checkpoint

默认：

```text
checkpoints_path = None
```

即不保存最终 checkpoint。

如果需要：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --checkpoints_path="checkpoints/OPT" --device="cuda:0"
```

最终保存：

```text
final.pt
```

其中包含：

```text
learner
config
state_mean
state_std
reward_info
```

`learner` 中包括：

```text
actor
actor_target

offline critics
online critics
target critics

density-ratio network

optimizers

SPOT VAE（如果使用）
```

---

## 23. 快速运行测试

正式跑 1M 实验之前建议先做短测试。

### MuJoCo

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --offline_steps=1000 --online_steps=30000 --eval_every=5000 --eval_episodes=3 --device="cuda:0"
```

注意：

```text
n_tau = 25000
```

因此如果：

```text
online_steps < 25000
```

根本不会进入 OPT Online Pre-Training 和后续 online fine-tuning。

短测试建议：

```text
online_steps >= 30000
```

---

### AntMaze

```bash
python opt.py --env_name="antmaze-umaze-v2" --train_seed=0 --offline_steps=1000 --online_steps=30000 --eval_every=5000 --eval_episodes=3 --vae_steps=1000 --device="cuda:0"
```

---

### Adroit

```bash
python opt.py --env_name="pen-cloned-v1" --train_seed=0 --offline_steps=1000 --online_steps=30000 --eval_every=5000 --eval_episodes=3 --vae_steps=1000 --device="cuda:0"
```

短测试只用于检查：

```text
D4RL environment 是否正常
dataset 是否能加载
offline training 是否正常
SPOT VAE 是否正常
25000-step collection 是否正常
Online Pre-Training 是否启动
HotPlug backward 是否报错
balanced replay 是否正常
TensorBoard 是否记录
GPU 是否正常
```

短测试结果不能作为正式论文结果。

---

## 24. 手动关闭环境自动配置

仅用于消融或代码调试：

```bash
python opt.py --env_name="hopper-medium-v2" --train_seed=0 --auto_env_config=False
```

此时：

```text
config_source=official:manual_override
```

或者：

```text
config_source=adapted:manual_override
```

正式 OPT baseline 不建议关闭。

否则你需要自行保证：

```text
TD3 / SPOT backbone
kappa
UTD
LayerNorm
reward normalization
support lambda
online discount
```

全部正确。

---

## 25. 论文中如何标注

主结果表可以写：

```text
OPT
```

对于官方覆盖环境，建议在实验设置中说明：

```text
We implement OPT following the original paper and the authors'
official PyTorch implementation, using the released task-specific
hyperparameters.
```

对于本项目额外加入的 MuJoCo medium-expert 和 Adroit human/expert：

```text
For D4RL tasks not evaluated in the original OPT study, we apply the
same OPT algorithm using a fixed same-domain hyperparameter transfer,
without task-specific tuning.
```

不要写：

```text
official OPT results
```

因为你跑的是自己重新执行的 PyTorch reproduction。

更准确的说法是：

```text
OPT (our reproduction)
```

或者主表写：

```text
OPT
```

并在 Appendix 中统一说明实现来源。

---

## 26. 哪些结果可以称为 OPT

### 官方 19 个任务

可以直接称为：

```text
OPT
```

只要没有关闭：

```text
auto_env_config
```

并且没有自行改动核心算法。

---

### 额外 11 个 RLPD 对齐任务

也可以在统一 benchmark 表格中称为：

```text
OPT
```

因为没有修改 OPT objective 或训练流程，只是将同一个算法应用到原论文没有覆盖的 D4RL dataset variant。

但 Appendix 必须说明：

```text
fixed same-domain hyperparameter transfer
```

不能声称这些是论文作者原本报告的任务。

---

## 27. 正式实验建议

正式生成论文 baseline 结果前固定：

```text
offline_steps
online_steps
eval_every
eval_episodes
train seeds
eval seed policy
```

环境相关参数：

```text
backbone
kappa schedule
UTD
SPOT lambda
LayerNorm
reward normalization
online discount
```

应由当前 `auto_env_config` 固定，不要根据最终结果逐环境再调。

推荐主实验：

```text
5 independent training seeds
```

报告：

```text
mean ± standard deviation
```

如果你的论文统一使用：

```text
1M online interactions
```

则所有 OPT、RLPD、WSRL、Cal-QL 和你的方法都应在同一 interaction budget 下比较。

---

## 28. 推荐工作流

正式实验前：

```text
1. MuJoCo 先测试 hopper-medium-v2
2. AntMaze 先测试 antmaze-umaze-v2
3. Adroit 先测试 pen-cloned-v1
4. 确认 Online Pre-Training 在 step 25000 后正常执行
5. 确认 evaluation/normalized_score 正常记录
6. 再开始完整 1M-step 实验
7. 每个环境完成 5 seeds 后统一统计 mean ± std
```

---

## 29. 依赖

主要依赖：

```text
Python
PyTorch
Gym
D4RL
NumPy
pyrallis
TensorBoard
tqdm
MuJoCo
```

Adroit 环境还需要对应的 D4RL / MuJoCo hand 环境依赖。

如果当前环境已经能运行项目里的 RLPD D4RL 实验，建议尽量复用该环境，不要为了 OPT 单独升级 Gym / D4RL / MuJoCo。

---

## 30. 最重要的注意事项

正式实验时最容易出错的是以下几点：

```text
1. 不要关闭 auto_env_config。
2. 不要把所有环境都当成 TD3+OPT。
3. AntMaze 和 Adroit 必须走 SPOT+OPT。
4. online_steps 必须大于 N_tau=25000，否则 OPT 核心 online pre-training 根本不会发生。
5. medium、medium-replay、random 的 kappa schedule 不相同。
6. MuJoCo medium-expert 和 Adroit human/expert 是 benchmark adaptation，不是 OPT 原论文正式任务。
7. 统一论文比较时必须使用相同 online interaction budget。
8. 短测试结果不能用于论文。
