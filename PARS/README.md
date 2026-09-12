# PARS PyTorch

`pars.py` 是 **PARS (Penalizing Infeasible Actions and Reward Scaling)** 的单文件 PyTorch 实现，用于 Offline-to-Online Reinforcement Learning（O2O RL）实验。

算法依据：

- Jeonghye Kim et al., **Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data**, ICML 2025 Spotlight.
- 作者官方代码：`https://github.com/LGAI-Research/pars`

当前实现的算法逻辑以 **PARS 论文 + 作者官方 JAX 代码**为准，并针对本项目的统一 RLPD benchmark 提供 30 个 D4RL 环境的 `env_name` 自动参数配置。

---

## 1. 运行方式

最简单的运行方式：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0
```

指定 GPU：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --device="cuda:0"
```

AntMaze：

```bash
python pars.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

Adroit：

```bash
python pars.py --env_name="pen-cloned-v1" --train_seed=0 --device="cuda:0"
```

只需要修改 `env_name`，代码会自动加载对应的 PARS 算法参数。

训练步数、评估频率、评估回合数、随机种子和 GPU 等实验协议参数仍由用户控制。

---

# 2. 支持的 30 个环境

## 2.1 MuJoCo

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

## 2.2 AntMaze

```text
antmaze-umaze-v2
antmaze-umaze-diverse-v2
antmaze-medium-play-v2
antmaze-medium-diverse-v2
antmaze-large-play-v2
antmaze-large-diverse-v2
```

## 2.3 Adroit

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

# 3. PARS 的核心机制

PARS 主要由两个机制组成：

```text
RS-LN
+
PA
```

即：

```text
Reward Scaling + Layer Normalization
+
Penalizing Infeasible Actions
```

普通 offline RL 的 Q 网络可能在数据分布之外产生线性外推，从而给不可行动作异常高的 Q 值。

PARS 的思路是：

1. 通过 Reward Scaling 扩大有效价值尺度；
2. 在 critic 中使用 LayerNorm，提高特征表达能力；
3. 人工构造明显位于可行动作区域之外的 infeasible actions；
4. 强制这些动作的 Q 值接近任务的最低合理价值 `Q_min`。

critic loss 可以概括为：

```math
L_Q
=
L_TD
+
alpha * L_PA
```

其中：

```math
L_PA
=
E[(Q(s,a_ood) - Q_min)^2].
```

因此 PARS 不是普通 TD3+BC，也不是只进行 reward scaling 的 TD3。

---

# 4. Infeasible Action Penalty

官方实现从动作空间 `[-1,1]` 外部构造动作。

首先：

```text
u ~ Uniform(-1, 1)
```

然后：

```text
u < 0  -> u - 1
u >= 0 -> u + 1
```

因此每个动作维度进入：

```text
[-2,-1] U [1,2]
```

再乘以 infeasible-region distance。

离线训练中：

```text
AntMaze: distance = 1000
其他环境: distance = 100
```

这些动作显然远离合法动作区域。

PARS 将其 Q 值压向：

```text
Q_min
```

---

# 5. Q_min

按照 PARS 论文 Appendix：

```text
halfcheetah : -366
hopper      : -166
walker2d    : -229

pen         : -715
door        : -42
hammer      : -348
relocate    : 0

antmaze     : 0
```

代码会根据 `env_name` 自动选择。

不要自行统一设置：

```text
Q_min = 0
```

因为这会改变 PARS 的 PA 机制。

---

# 6. Reward Scaling

Reward Scaling 是 PARS 的核心组成部分。

## MuJoCo

```text
HalfCheetah: reward_scale = 5
Hopper:      reward_scale = 10
Walker2d:    reward_scale = 10
```

## Adroit

```text
reward_scale = 10
```

## AntMaze

逐环境配置：

| Environment | Reward Scale |
|---|---:|
| antmaze-umaze-v2 | 10000 |
| antmaze-umaze-diverse-v2 | 10000 |
| antmaze-medium-play-v2 | 1000 |
| antmaze-medium-diverse-v2 | 1000 |
| antmaze-large-play-v2 | 1000 |
| antmaze-large-diverse-v2 | 10000 |

在线采集的数据使用与 offline dataset 相同的 reward scale。

---

# 7. LayerNorm

PARS critic 默认：

```text
layer_norm = True
```

LayerNorm 与 Reward Scaling 一起构成论文中的：

```text
RS-LN
```

因此正式 PARS baseline 不应关闭 LayerNorm。

---

# 8. Critic Ensemble

PARS 使用 ensemble critic。

## MuJoCo / Adroit

默认：

```text
num_ensemble = 10
```

## AntMaze

根据作者公开 AntMaze 配置：

```text
num_ensemble = 4
critic_hidden_layers = 3
```

其他任务默认：

```text
critic_hidden_layers = 2
```

---

# 9. Offline actor objective

离线 actor 使用：

```math
L_actor
=
beta * ||pi(s)-a_D||^2
-
lambda * Q(s,pi(s))
```

其中：

```math
lambda
=
1 / mean(|Q|)
```

在 offline 阶段：

```text
Q(s,pi(s))
```

使用 critic ensemble 中的最小值。

`beta` 控制 behavior-cloning regularization。

---

# 10. Online actor objective

在线阶段 actor 仍然使用：

```math
L_actor
=
beta_online * ||pi(s)-a||^2
-
lambda * Q(s,pi(s)).
```

但是 Q 不再固定使用整个 ensemble 的最小值，而是根据任务配置随机选择一定数量 critic 后取均值。

代码中的参数：

```text
online_num_action_sample_ensemble
```

对应论文中的：

```text
S_kactor
```

---

# 11. 一个重要的官方代码差异

PARS 论文和作者发布的 YAML 配置均显式给出了：

```text
online_beta
```

但公开 JAX 代码的 `online_train()` 最终将：

```text
self.beta
```

传给 online actor loss，而不是：

```text
self.online_beta
```

这导致 YAML 中的 `online_beta` 实际不会生效。

本实现按照：

```text
论文 > 存在冲突的代码实现
```

的原则，在线 actor 使用：

```text
online_beta
```

这与论文 Table 10 及作者配置文件的实验意图一致。

因此本实现是：

```text
paper-faithful PARS reproduction
```

而不是逐行保留该疑似实现疏漏。

---

# 12. PARS 通用超参数

论文给出的通用参数：

```text
optimizer              = Adam
batch_size             = 256
learning_rate          = 3e-4
tau                    = 0.005
hidden_dim             = 256

gamma:
    AntMaze             = 0.995
    others              = 0.99

infeasible distance:
    AntMaze offline     = 1000
    others offline      = 100

actor cosine scheduler:
    Adroit              = True
    others              = False

online UTD             = 20
learning_starts        = 0
```

在线 exploration noise：

```text
MuJoCo = 0.1
AntMaze = 0.05
Adroit = 0.05
```

---

# 13. MuJoCo 自动参数

以下参数直接来自 PARS 论文 Table 11。

## HalfCheetah

| Dataset | Offline α | Policy Noise | Offline S_kcritic | Online α | Online S_kactor |
|---|---:|---:|---:|---:|---:|
| random | 0.0001 | 0.2 | 2 | 0.0001 | 1 |
| medium | 0.0001 | 0 | 2 | 0.0001 | 10 |
| medium-replay | 0.0001 | 0 | 2 | 0.0001 | 10 |
| medium-expert | 0.0001 | 0.2 | 10 | adapted | adapted |

HalfCheetah online replay：

```text
offline_ratio = 0.05
```

即：

```text
5% offline
95% online
```

---

## Hopper

| Dataset | Offline α | Policy Noise | Offline S_kcritic | Online α | Online S_kactor |
|---|---:|---:|---:|---:|---:|
| random | 0.01 | 0.2 | 2 | 0.01 | 1 |
| medium | 0.01 | 0 | 10 | 0.1 | 1 |
| medium-replay | 0.01 | 0 | 10 | 0.1 | 1 |
| medium-expert | 0.0001 | 0.2 | 10 | adapted | adapted |

在线 replay：

```text
random:
    offline_ratio = 0.05

medium / medium-replay:
    offline_ratio = 0.5
```

---

## Walker2d

| Dataset | Offline α | Policy Noise | Offline S_kcritic | Online α | Online S_kactor |
|---|---:|---:|---:|---:|---:|
| random | 0.01 | 0 | 10 | 0.0001 | 10 |
| medium | 0.01 | 0 | 10 | 0.1 | 1 |
| medium-replay | 0.01 | 0 | 10 | 0.01 | 1 |
| medium-expert | 0.0001 | 0.2 | 10 | adapted | adapted |

---

# 14. MuJoCo medium-expert 的处理

PARS 原论文报告了 medium-expert 的 offline 设置，但没有在 online-finetuning Table 11 中给出 online 参数。

因此：

```text
halfcheetah-medium-expert-v2
hopper-medium-expert-v2
walker2d-medium-expert-v2
```

在本项目中属于：

```text
config_source=adapted
```

固定规则：

```text
offline 部分：
    使用论文 medium-expert 的官方参数

online 部分：
    使用同一环境 medium dataset 的 online α 和 S_kactor
```

例如：

```text
hopper-medium-expert
```

offline：

```text
alpha = 0.0001
policy_noise = 0.2
S_kcritic = 10
```

online：

```text
继承 hopper-medium:
alpha = 0.1
S_kactor = 1
```

不进行 medium-expert 单独调参。

---

# 15. AntMaze 自动参数

PARS 论文 Table 10：

| Environment | reward scale | Offline β | Offline α | Online β | Online α |
|---|---:|---:|---:|---:|---:|
| umaze | 10000 | 0.005 | 0.001 | 0 | 0.001 |
| umaze-diverse | 10000 | 0.005 | 0.001 | 0.001 | 0.001 |
| medium-play | 1000 | 0.01 | 0.001 | 0 | 0.001 |
| medium-diverse | 1000 | 0.01 | 0.001 | 0 | 0.001 |
| large-play | 1000 | 0.01 | 0.001 | 0.01 | 0.001 |
| large-diverse | 10000 | 0.01 | 0.01 | 0.01 | 0.001 |

其他 AntMaze 参数：

```text
gamma = 0.995

num_ensemble = 4
num_sample_ensemble = 2
online_num_sample_ensemble = 2
online_num_action_sample_ensemble = 1

critic_hidden_layers = 3

offline_ratio = 0.5
UTD = 20

exploration_noise = 0.05

offline_ood_action_weight = 1000
online_ood_action_weight = 100
```

---

# 16. Adroit 自动参数

所有 Adroit：

```text
reward_scale = 10
gamma = 0.99

num_ensemble = 10
num_sample_ensemble = 2
online_num_sample_ensemble = 2
online_num_action_sample_ensemble = 1

offline_ratio = 0.5
UTD = 20

exploration_noise = 0.05

actor_cosine_scheduler = True
```

## Cloned

| Environment | Offline β | Offline α | Online β | Online α |
|---|---:|---:|---:|---:|
| pen-cloned | 0.01 | 0.01 | 0 | 0.001 |
| door-cloned | 0.01 | 0.01 | 0.01 | 0.001 |
| hammer-cloned | 0.1 | 0.001 | 0 | 0.001 |
| relocate-cloned | 0.01 | 0.01 | 0.01 | 0.001 |

这四个属于 PARS 官方 O2O benchmark。

---

# 17. Adroit expert

PARS 论文给出了 expert 的 offline 参数：

| Environment | Offline β | Offline α |
|---|---:|---:|
| pen-expert | 0.01 | 0.01 |
| door-expert | 0.1 | 0.001 |
| hammer-expert | 0.01 | 0.001 |
| relocate-expert | 0.1 | 0.001 |

但没有报告 expert 的 online-finetuning 配置。

因此 expert 在本项目中：

```text
config_source=adapted
```

处理规则：

```text
offline:
    使用论文 expert 官方参数

online:
    使用同一 task 的 cloned online 参数
```

例如：

```text
door-expert-v1
```

online：

```text
online_beta = door-cloned 的 0.01
online_alpha = 0.001
```

---

# 18. Adroit human

PARS 原论文没有报告 standard D4RL Adroit human 的 PARS 配置。

为了与 RLPD benchmark 对齐：

```text
pen-human-v1
door-human-v1
hammer-human-v1
relocate-human-v1
```

使用固定同任务 cloned 参数：

```text
human -> same-task cloned
```

例如：

```text
hammer-human-v1
```

使用：

```text
offline_beta = 0.1
offline_alpha = 0.001

online_beta = 0
online_alpha = 0.001
```

启动时明确打印：

```text
config_source=adapted
```

论文中必须说明这属于 benchmark adaptation。

---

# 19. 官方 O2O 环境与适配环境

## 官方 PARS online-finetuning 环境

共 19 个：

```text
MuJoCo:
halfcheetah-random-v2
halfcheetah-medium-v2
halfcheetah-medium-replay-v2
hopper-random-v2
hopper-medium-v2
hopper-medium-replay-v2
walker2d-random-v2
walker2d-medium-v2
walker2d-medium-replay-v2

AntMaze:
6 个全部

Adroit:
pen-cloned-v1
door-cloned-v1
hammer-cloned-v1
relocate-cloned-v1
```

程序显示：

```text
config_source=official
```

---

## 适配到统一 RLPD benchmark 的 11 个环境

```text
MuJoCo medium-expert:
3 个

Adroit expert:
4 个

Adroit human:
4 个
```

程序显示：

```text
config_source=adapted
```

---

# 20. strict_official_only

如果只允许论文正式 O2O 环境：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --strict_official_only=True
```

如果运行：

```bash
python pars.py --env_name="hopper-medium-expert-v2" --train_seed=0 --strict_official_only=True
```

程序会报错。

本项目为了覆盖完整 30 个 RLPD 环境时保持默认：

```text
strict_official_only = False
```

---

# 21. Online Replay Mixing

PARS 在线阶段同时使用：

```text
offline replay
+
online replay
```

### AntMaze

```text
50% offline
50% online
```

### Adroit

```text
50% offline
50% online
```

### MuJoCo

论文规定：

```text
HalfCheetah:
5% offline + 95% online

random datasets:
5% offline + 95% online

其他 MuJoCo:
50% offline + 50% online
```

代码自动处理。

---

# 22. UTD

所有在线 PARS 实验：

```text
utd_ratio = 20
```

即每获得：

```text
1 online transition
```

执行：

```text
20 critic gradient updates
```

随后在线 actor 更新一次，并进行 target-network soft update。

这与作者公开代码的在线循环结构一致。

---

# 23. Offline 与 Online 的 target critic

Offline：

```text
每 2 次训练 iteration 更新一次 actor
```

因为：

```text
policy_frequency = 2
```

target actor 和 target critic 与 actor 一起更新。

Online：

作者代码进入 fine-tuning 后将：

```text
policy_frequency = 1
```

所以每一个 online environment step：

```text
20 critic updates
+
1 actor update
+
1 target update
```

---

# 24. 实验协议参数

以下参数不会因为 `env_name` 自动改变：

```text
pretrain_steps
num_total_steps
eval_every
eval_episodes
log_every

train_seed
eval_seed

device
checkpoints_path
```

当前默认：

```text
pretrain_steps = 3,000,000
num_total_steps = 1,000,000

eval_every = 5,000
eval_episodes = 10

batch_size = 256
```

其中：

```text
3M offline updates
```

来自 PARS 官方训练程序默认协议。

```text
1M online steps
```

用于与你当前统一 O2O benchmark 对齐。

---

# 25. 原论文 online interaction budget

PARS 官方 `main.py` 默认：

```text
online_max_timesteps = 300000
```

如果要复现论文原始 online budget，可以：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --num_total_steps=300000 --device="cuda:0"
```

如果你的论文要求所有 baseline 都使用统一：

```text
1M online interactions
```

则使用：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --num_total_steps=1000000 --device="cuda:0"
```

不要在同一张表中混合 300k 和 1M 而不说明。

---

# 26. 推荐正式实验命令

## MuJoCo

```bash
python pars.py --env_name="halfcheetah-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="hopper-medium-replay-v2" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="walker2d-medium-expert-v2" --train_seed=0 --device="cuda:0"
```

## AntMaze

```bash
python pars.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="antmaze-medium-play-v2" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="antmaze-large-diverse-v2" --train_seed=0 --device="cuda:0"
```

## Adroit

```bash
python pars.py --env_name="pen-cloned-v1" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="door-human-v1" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="hammer-expert-v1" --train_seed=0 --device="cuda:0"
```

---

# 27. 五个随机种子

例如：

```text
0
10
20
30
40
```

分别运行：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=10 --device="cuda:0"
```

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=20 --device="cuda:0"
```

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=30 --device="cuda:0"
```

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=40 --device="cuda:0"
```

---

# 28. TensorBoard

默认：

```text
log_root = logs/PARS
```

日志目录：

```text
logs/PARS/<domain>/<env_name>_seed<seed>_PARS/
```

例如：

```text
logs/PARS/hopper/hopper-medium-v2_seed0_PARS/
```

启动：

```bash
tensorboard --logdir logs/PARS --port 8088
```

---

# 29. TensorBoard 指标

Offline：

```text
offline/critic_loss
offline/q_loss
offline/q_ood_loss
offline/q
offline/q_ood

offline/actor_loss
offline/actor_q
offline/bc_penalty
offline/actor_q_scale

offline_evaluation/return
offline_evaluation/d4rl_normalized_score
```

Online：

```text
online/critic_loss
online/q_loss
online/q_ood_loss
online/q
online/q_ood

online/actor_loss
online/actor_q
online/bc_penalty
online/actor_q_scale

online/offline_ratio
online/utd_ratio

online_evaluation/return
online_evaluation/d4rl_normalized_score
```

正式论文曲线通常读取：

```text
online_evaluation/d4rl_normalized_score
```

---

# 30. Checkpoint

默认：

```text
checkpoints_path = None
```

如需保存：

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --checkpoints_path="checkpoints/PARS" --device="cuda:0"
```

最终保存：

```text
final.pt
```

包括：

```text
actor
actor_target
critic ensemble
critic target ensemble
optimizers
config
state normalization statistics
```

---

# 31. 快速测试

由于正式 PARS：

```text
offline = 3M updates
online UTD = 20
```

计算量较大。

正式运行前建议短测试。

## MuJoCo

```bash
python pars.py --env_name="hopper-medium-v2" --train_seed=0 --pretrain_steps=1000 --num_total_steps=5000 --eval_every=1000 --eval_episodes=3 --device="cuda:0"
```

## AntMaze

```bash
python pars.py --env_name="antmaze-umaze-v2" --train_seed=0 --pretrain_steps=1000 --num_total_steps=5000 --eval_every=1000 --eval_episodes=3 --device="cuda:0"
```

## Adroit

```bash
python pars.py --env_name="pen-cloned-v1" --train_seed=0 --pretrain_steps=1000 --num_total_steps=5000 --eval_every=1000 --eval_episodes=3 --device="cuda:0"
```

短测试仅用于检查：

```text
D4RL dataset loading
reward scaling
ensemble critic
PA loss
offline actor
online mixed replay
UTD=20
TensorBoard
GPU
```

短测试结果不能用于论文。

---

# 32. 运行时应该检查什么

程序启动后会打印类似：

```text
env=hopper-medium-v2
config_source=official
reward_scale=10
q_min=-166
offline_alpha=0.01
offline_beta=0
online_alpha=0.1
online_beta=0
ensemble=10
offline_target_samples=10
online_target_samples=2
online_actor_samples=1
offline_ratio=0.5
UTD=20
```

在正式长实验前建议核对这些值。

对于适配环境会打印：

```text
config_source=adapted
```

并出现提示。

---

# 33. 论文中如何写

主表可以直接写：

```text
PARS
```

对于原论文正式在线任务：

```text
We implement PARS following the original paper and the authors'
official JAX implementation, using the reported task-specific
hyperparameters.
```

对于额外 11 个 RLPD 对齐环境：

```text
For D4RL tasks not evaluated in the original PARS online-finetuning
benchmark, we use a fixed same-domain hyperparameter transfer without
task-specific tuning.
```

---

# 34. 哪些结果可以说是 PARS

## 官方 19 个 O2O 环境

可以直接称：

```text
PARS
```

前提是：

```text
auto_env_config=True
```

并且没有修改核心算法。

---

## 11 个适配环境

也可以在统一 benchmark 中标：

```text
PARS
```

因为：

```text
RS-LN
PA
ensemble critic
actor objective
mixed replay
UTD protocol
```

都没有改变。

但 Appendix 必须说明：

```text
same-domain hyperparameter transfer
```

不能声称：

```text
officially reported PARS setting
```

---

# 35. 不建议修改的参数

如果目的是正式 baseline，不建议手动改：

```text
reward_scale
q_min

offline_alpha
offline_beta
online_alpha
online_beta

offline_ratio

num_ensemble
num_sample_ensemble
online_num_action_sample_ensemble

critic_hidden_layers
layer_norm

discount
offline_ood_action_weight
```

这些属于算法/环境配置。

---

# 36. 可以统一控制的实验参数

适合根据你的整篇论文统一：

```text
pretrain_steps
num_total_steps
eval_every
eval_episodes
train_seed
eval_seed
device
checkpoints_path
log_every
```

例如你的统一协议：

```text
online interactions = 1,000,000
eval every = 5,000
eval episodes = 10
5 seeds = 0,10,20,30,40
```

可以直接保持。

---

# 37. 依赖

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

如果现有环境已经能够运行 RLPD 的 D4RL 实验，建议复用该环境。

不要为了 PARS 单独随意升级：

```text
Gym
D4RL
MuJoCo
```

因为 D4RL 老版本 API 容易出现兼容问题。

---

# 38. 正式实验注意事项

最重要的几点：

```text
1. PARS 必须同时包含 RS-LN 和 PA。
2. 不要关闭 critic LayerNorm。
3. reward_scale 必须随任务正确变化。
4. Q_min 不应统一设置为 0。
5. AntMaze 使用 gamma=0.995。
6. 在线 UTD=20，计算量明显高于普通 TD3。
7. MuJoCo 的 replay offline ratio 并不统一。
8. online beta 应使用论文/YAML 定义值。
9. medium-expert、Adroit expert/human 属于 O2O benchmark adaptation。
10. 正式比较时确保所有方法使用相同 online interaction budget。
```

---

# 39. 推荐实验流程

```text
1. hopper-medium-v2 短测试
2. antmaze-umaze-v2 短测试
3. pen-cloned-v1 短测试
4. 检查 q_ood 是否逐步向 Q_min 靠近
5. 检查 normalized score 正常记录
6. 再运行完整 3M offline + 1M online
7. 每个环境完成 5 seeds
8. 汇总 mean ± std
```

---

## References

PARS paper:

```text
Jeonghye Kim, Yongjae Shin, Whiyoung Jung, Sunghoon Hong,
Deunsol Yoon, Youngchul Sung, Kanghoon Lee, Woohyung Lim.
Penalizing Infeasible Actions and Reward Scaling in Reinforcement
Learning with Offline Data.
ICML 2025.
```

Official implementation:

```text
https://github.com/LGAI-Research/pars
```
