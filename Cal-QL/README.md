# Cal-QL PyTorch

本目录中的 `calql.py` 是用于 Offline-to-Online Reinforcement Learning 实验的单文件 PyTorch 实现。

实现依据：

- Cal-QL 论文：**Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning**, NeurIPS 2023。
- 作者官方 JAX/Flax 实现：`nakamotoo/Cal-QL`。
- `Hsu_o2o_rl` 仓库仅用于参考单文件组织、命令行和日志风格，不作为算法实现依据。

本实现的目标是直接支持与本项目 RLPD 实验对应的 30 个 D4RL 环境，并根据 `env_name` 自动配置环境相关的 Cal-QL 参数。

---

## 1. 基本运行方式

最简单的运行方式：

```bash
python calql.py --env_name="antmaze-umaze-v2" --train_seed=0
```

指定 GPU：

```bash
python calql.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

MuJoCo 示例：

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=0 --device="cuda:0"
```

Adroit 示例：

```bash
python calql.py --env_name="pen-human-v1" --train_seed=0 --device="cuda:0"
```

`env_name` 决定环境相关算法配置；训练步数、评估频率、随机种子、UTD、batch size 等实验协议参数仍由用户控制。

---

## 2. 支持的环境

`calql.py` 直接支持与本项目 RLPD 对应的 30 个环境。

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

### AntMaze

```text
antmaze-umaze-v2
antmaze-umaze-diverse-v2
antmaze-medium-play-v2
antmaze-medium-diverse-v2
antmaze-large-play-v2
antmaze-large-diverse-v2
```

其他环境默认拒绝运行，避免在没有对应实验依据的情况下静默套用错误超参数。

---

## 3. Cal-QL 核心机制

Cal-QL 基于 SAC + CQL。

普通 CQL 的 conservative regularizer 会对策略采样动作的 Q 值进行保守约束。Cal-QL 的核心修改是使用参考策略价值对这些 Q 值进行校准：

```text
Q_calibrated(s,a) = max(Q(s,a), V_ref(s))
```

代码中仅对 conservative regularizer 内的 policy-sampled Q values 进行 lower-bound calibration。

random-action Q 和 dataset-action Q 不会被错误地执行相同的 clamp。

因此，本实现保留了 Cal-QL 的核心算法目标，而不是简单的 CQL 微调。

---

## 4. Reference Value 的构造

不同环境使用不同的 reference estimator。

### 4.1 AntMaze

AntMaze 使用：

```text
reference_mode = mc
```

即使用行为轨迹的 Monte-Carlo return-to-go 作为 reference value。

这是作者公开 AntMaze 实现所采用的方式。

奖励自动转换为：

```text
r_calql = 10 * r - 5
```

因此原始 AntMaze：

```text
failure reward = 0
success reward = 1
```

变为：

```text
failure reward = -5
success reward = 5
```

---

### 4.2 MuJoCo Locomotion

包括：

```text
HalfCheetah
Hopper
Walker2d
```

使用：

```text
reference_mode = sarsa
```

原因是 D4RL locomotion 数据通常由 time-limit 截断，并不都是真正的终止轨迹，因此不能直接把截断后的 Monte-Carlo return 当作可靠的行为策略价值。

按照 Cal-QL 论文 Appendix D，本实现首先在固定 offline dataset 上拟合一个 SARSA reference Q：

```text
Q_ref(s_t,a_t)
    <- r_t + gamma * Q_ref_target(s_{t+1},a_{t+1})
```

然后使用：

```text
Q_ref(s,a)
```

作为 offline Cal-QL calibration reference。

需要注意：

Cal-QL 论文明确说明 locomotion 使用 fitted SARSA Q-function，但作者没有公开完整的 locomotion reference-fitting 实现和全部辅助超参数。

因此：

```text
reference_train_steps
reference_batch_size
reference_learning_rate
reference_tau
reference_target_update_period
```

属于显式实验参数，不应被描述为作者官方逐项公布的超参数。

---

### 4.3 标准 D4RL Adroit

本项目使用的是：

```text
human
cloned
expert
```

例如：

```text
pen-human-v1
door-cloned-v1
hammer-expert-v1
```

原 Cal-QL 论文主要使用的是 Adroit-Binary benchmark，并不是上述标准 D4RL human/cloned/expert 数据集。

为了和 RLPD benchmark 完全对应，本实现：

```text
reference_mode = sarsa
```

即保留 Cal-QL 原始 objective，同时使用 fitted-SARSA reference estimator。

标准 D4RL Adroit 的 dense reward 保持不变：

```text
reward_scale = 1
reward_bias = 0
```

不会错误套用原论文 Adroit-Binary 的 reward transform。

因此论文中应将这部分理解为：

```text
Cal-QL applied to the standard D4RL Adroit benchmark
```

而不是声称它是作者官方公开的 Adroit-Binary 实验配置。

---

## 5. 环境相关参数自动配置

只需要提供：

```text
--env_name
```

代码会自动识别环境类别。

### AntMaze

自动配置：

```text
actor_hidden_dims     = (256, 256)
critic_hidden_dims    = (256, 256, 256, 256)
reference_hidden_dims = critic_hidden_dims

reference_mode        = mc

cql_min_q_weight      = 5.0
cql_lagrange          = True
cql_target_action_gap = 0.8

max_target_backup     = True
target_backup_n_actions = 10

reward_scale          = 10.0
reward_bias           = -5.0
```

---

### MuJoCo Locomotion

自动配置：

```text
actor_hidden_dims     = (256, 256)
critic_hidden_dims    = (256, 256)
reference_hidden_dims = (256, 256)

reference_mode        = sarsa

cql_min_q_weight      = 5.0
cql_lagrange          = False
cql_target_action_gap = 1.0

max_target_backup     = True

reward_scale          = 1.0
reward_bias           = 0.0
```

其中 fitted-SARSA 的训练超参数并没有被伪装成环境官方参数，而是保留为实验参数。

---

### 标准 D4RL Adroit

自动配置：

```text
actor_hidden_dims     = (512, 512)
critic_hidden_dims    = (512, 512, 512)
reference_hidden_dims = (512, 512, 512)

reference_mode        = sarsa

cql_min_q_weight      = 1.0
cql_lagrange          = False
cql_target_action_gap = 1.0

max_target_backup     = True

reward_scale          = 1.0
reward_bias           = 0.0
```

---

## 6. 实验协议参数

以下参数不会因为环境不同而自动修改。

当前默认值：

```text
pretrain_steps       = 1,000,000
num_total_steps      = 1,000,000
batch_size           = 256
online_utd_ratio     = 1
mixing_ratio         = 0.5
replay_buffer_size   = 1,000,000

eval_every           = 5,000
eval_episodes        = 10
log_every            = 1,000
```

这些参数属于论文实验协议。

如果论文要求所有 baseline 使用统一的：

```text
1M online environment steps
evaluation every 5,000 steps
10 evaluation episodes
```

则当前默认值已经符合这一协议。

仍建议在正式实验命令中明确指定关键参数，避免后续修改代码默认值后造成实验协议不一致。

例如：

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=0 --num_total_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
```

---

## 7. SARSA Reference 参数

MuJoCo 和标准 D4RL Adroit 会先训练 fitted-SARSA reference Q。

当前默认：

```text
reference_train_steps          = 1,000,000
reference_batch_size           = 256
reference_learning_rate        = 3e-4
reference_tau                  = 0.005
reference_target_update_period = 1
```

例如修改 reference fitting 步数：

```bash
python calql.py --env_name="walker2d-medium-v2" --train_seed=0 --reference_train_steps=500000 --device="cuda:0"
```

注意：

这些参数会影响 Cal-QL 的 reference quality。

正式论文实验中，一旦确定，应对所有同类别环境保持预先定义的统一规则，不建议根据最终结果逐环境调参。

---

## 8. Offline-to-Online 训练流程

程序执行顺序如下。

### AntMaze

```text
读取 D4RL offline dataset
        ↓
计算 behavior trajectory MC return-to-go
        ↓
Cal-QL offline pre-training
        ↓
online interaction
        ↓
完整轨迹写入 online replay buffer
        ↓
offline + online mixed replay
        ↓
Cal-QL online fine-tuning
```

### MuJoCo / 标准 D4RL Adroit

```text
读取 D4RL offline dataset
        ↓
构造 SARSA tuples
        ↓
训练 fitted-SARSA reference Q
        ↓
计算 offline reference Q values
        ↓
Cal-QL offline pre-training
        ↓
online interaction
        ↓
offline + online mixed replay
        ↓
Cal-QL online fine-tuning
```

---

## 9. Offline / Online Replay 混合

在线阶段默认：

```text
mixing_ratio = 0.5
```

表示每个在线训练 batch 中：

```text
50% offline dataset
50% online replay buffer
```

例如 batch size 为 256，则通常为：

```text
128 offline transitions
128 online transitions
```

可通过：

```text
--mixing_ratio
```

修改。

例如：

```bash
python calql.py --env_name="antmaze-medium-play-v2" --train_seed=0 --mixing_ratio=0.5
```

---

## 10. UTD

默认：

```text
online_utd_ratio = 1
```

含义是每收集 1 个 online environment transition，对应执行约 1 次 gradient update。

代码以完整轨迹为单位收集数据。

若一次轨迹收集：

```text
trajectory_length = L
```

则随后进行：

```text
L * online_utd_ratio
```

次更新。

例如：

```text
L = 1000
online_utd_ratio = 1
```

则执行：

```text
1000 gradient updates
```

---

## 11. 推荐正式实验命令

### AntMaze

```bash
python calql.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="antmaze-medium-play-v2" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="antmaze-large-diverse-v2" --train_seed=0 --device="cuda:0"
```

### MuJoCo

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="hopper-medium-replay-v2" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="walker2d-medium-expert-v2" --train_seed=0 --device="cuda:0"
```

### Adroit

```bash
python calql.py --env_name="pen-human-v1" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="door-cloned-v1" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="relocate-expert-v1" --train_seed=0 --device="cuda:0"
```

---

## 12. 多随机种子

推荐正式实验使用独立随机种子，例如：

```text
0
10
20
30
40
```

分别运行：

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=10 --device="cuda:0"
```

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=20 --device="cuda:0"
```

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=30 --device="cuda:0"
```

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=40 --device="cuda:0"
```

当没有显式提供：

```text
dataset_seed
```

时，offline dataset sampling RNG 默认跟随：

```text
train_seed
```

因此不同训练 seed 对应独立的数据采样随机流。

---

## 13. TensorBoard 日志

默认日志根目录：

```text
logs/CalQL/
```

具体格式：

```text
logs/CalQL/<env_name>/seed_<train_seed>/
```

例如：

```text
logs/CalQL/halfcheetah-medium-v2/seed_0/
```

启动 TensorBoard：

```bash
tensorboard --logdir logs/CalQL --port 8088
```

主要记录包括：

```text
reference-sarsa/*
offline-training/*
offline-evaluation/*
online-training/*
online-evaluation/*
online-collection/*
```

其中 D4RL normalized score：

```text
offline-evaluation/d4rl_normalized_score
online-evaluation/d4rl_normalized_score
```

正式论文结果通常优先读取：

```text
online-evaluation/d4rl_normalized_score
```

---

## 14. Checkpoint

默认：

```text
checkpoints_path = None
```

即不保存模型 checkpoint。

如需保存：

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=0 --checkpoints_path="checkpoints/CalQL" --device="cuda:0"
```

最终文件包含：

```text
actor
qf1
qf2
target_qf1
target_qf2
log_alpha
log_alpha_prime
config
environment_steps
online_gradient_steps
```

对于使用 fitted SARSA reference 的任务，还会保存：

```text
reference_q
```

---

## 15. 快速运行测试

正式跑 1M 实验前，建议先做短测试确认 D4RL、MuJoCo 和代码版本兼容。

### AntMaze

```bash
python calql.py --env_name="antmaze-umaze-v2" --train_seed=0 --pretrain_steps=1000 --num_total_steps=5000 --eval_every=1000 --eval_episodes=3 --device="cuda:0"
```

### MuJoCo

MuJoCo 还需要先测试 SARSA reference：

```bash
python calql.py --env_name="halfcheetah-medium-v2" --train_seed=0 --reference_train_steps=1000 --pretrain_steps=1000 --num_total_steps=5000 --eval_every=1000 --eval_episodes=3 --device="cuda:0"
```

### Adroit

```bash
python calql.py --env_name="pen-human-v1" --train_seed=0 --reference_train_steps=1000 --pretrain_steps=1000 --num_total_steps=5000 --eval_every=1000 --eval_episodes=3 --device="cuda:0"
```

短测试结果不能用于论文，只用于检查：

```text
环境是否能够创建
D4RL dataset 是否能够读取
SARSA reference 是否正常训练
offline Cal-QL 是否正常更新
online replay 是否正常工作
TensorBoard 是否正常记录
GPU 是否正常运行
```

---

## 16. 依赖

代码直接依赖：

```text
Python
PyTorch
Gym
D4RL
NumPy
pyrallis
TensorBoard
tqdm
```

以及运行对应 D4RL 环境所需的 MuJoCo / Adroit 依赖。

如果当前已经能够正常运行本项目中的 RLPD D4RL 实验，建议直接复用同一个环境，再补充缺少的：

```text
pyrallis
tensorboard
tqdm
```

不要仅为了 Cal-QL 随意升级 Gym、D4RL 或 MuJoCo，因为 D4RL 属于较老的软件栈，版本变化可能引起环境注册或 API 兼容问题。

---

## 17. 论文中如何描述

主结果表中可以写：

```text
Cal-QL
```

建议在 Experimental Setup 或 Appendix 中说明：

```text
We implement Cal-QL in PyTorch following the original paper and the
authors' released JAX implementation.
```

对于 MuJoCo：

```text
For D4RL locomotion, following Appendix D of Cal-QL, we fit a SARSA
Q-function on the offline dataset to estimate the reference behavior value.
```

对于标准 D4RL Adroit：

```text
The original Cal-QL experiments use the Adroit-Binary benchmark. To match
our RLPD benchmark, we apply the same Cal-QL objective to the standard D4RL
Adroit human/cloned/expert datasets and estimate the reference value with a
fitted SARSA critic.
```

不要描述为：

```text
official Cal-QL implementation
```

因为当前文件是根据论文和官方实现重新实现的 PyTorch reproduction，而不是直接运行作者原始 JAX 程序。

---

## 18. 结果能否作为 Cal-QL baseline

### AntMaze

可以直接作为：

```text
Cal-QL
```

baseline。

该部分最接近作者公开实现路径。

### MuJoCo

可以作为：

```text
Cal-QL
```

的论文复现结果。

但应注明 fitted-SARSA reference 是依据论文 Appendix D 实现，因为作者没有公开完整 locomotion reference-fitting 代码及其全部辅助超参数。

正式实验前建议先验证若干与原论文重叠的 MuJoCo 环境，确认性能量级和趋势合理。

### 标准 D4RL Adroit

可以作为 Cal-QL 在本项目统一 benchmark 上的结果，但必须在实验设置中说明：

```text
standard D4RL Adroit adaptation
```

因为原论文使用的是 Adroit-Binary benchmark。

---

## 19. 正式实验注意事项

正式生成论文数据前建议固定以下内容：

```text
online interaction budget
offline pre-training updates
evaluation interval
evaluation episodes
random seeds
batch size
offline/online replay ratio
UTD
SARSA reference fitting protocol
```

同一组正式实验开始后，不应根据单个环境的最终结果反复修改参数。

建议正式实验至少使用：

```text
5 independent seeds
```

并报告：

```text
mean ± standard deviation
```

如果使用统一 1M-step O2O protocol，则建议固定：

```text
num_total_steps = 1,000,000
eval_every      = 5,000
eval_episodes   = 10
```

---

## 20. 推荐工作流

正式实验前：

```text
1. 每一类环境先跑一个短测试
2. 检查 SARSA reference loss（MuJoCo / Adroit）
3. 检查 offline normalized score
4. 检查 online normalized score
5. 检查 TensorBoard step 轴
6. 再开始完整 1M-step 实验
7. 完成一个环境的多个 seeds 后统一计算 mean ± std
```

不要用短测试结果代替正式实验结果。
