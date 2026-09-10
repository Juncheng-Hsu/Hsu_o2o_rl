# WSRL PyTorch Reimplementation (`warl.py`)

本项目提供一个单文件 PyTorch 版本的 WSRL（Warm-Start Reinforcement Learning）实现，对齐论文：

> **WSRL: Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data**  
> ICLR 2025

实现参考 WSRL 论文及作者公开的 JAX/Flax 代码，并保留单文件训练形式，便于直接作为 Offline-to-Online RL baseline 使用。

> 论文中建议将本实现标注为 **WSRL (PyTorch reimplementation)**，而不是宣称为官方代码运行结果。

---

## 1. 核心流程

`warl.py` 实现以下 WSRL 流程：

1. 使用 CQL / Cal-QL 进行 offline pretraining，或加载已有兼容 checkpoint；
2. offline pretraining 完成后丢弃 offline dataset；
3. 创建空的 online replay buffer；
4. 使用冻结的 pretrained policy 收集 `warmup_steps=5000` 条 online transition，不执行梯度更新；
5. warmup 后仅使用 online replay buffer 进行 SAC 高 UTD 更新；
6. 默认使用 10 个 Q-functions；
7. 每个环境步执行 4 次 critic update，并执行 1 次 actor / temperature update。

默认 WSRL online 更新配置：

```text
num_critics = 10
online_num_min_qs = 2
warmup_steps = 5000
utd_ratio = 4
critic minibatch = 256
actor super-batch = 1024
actor_lr = 1e-4
critic_lr = 3e-4
temperature_lr = 1e-4
gamma = 0.99
tau = 0.005
backup_entropy = False
offline_data_ratio_online = 0.0
```

---

## 2. 文件

```text
warl.py      WSRL PyTorch 单文件实现
README.md    使用说明
```

---

## 3. 环境依赖

建议在已经能够正常运行 D4RL + MuJoCo 的 Python 环境中运行。

代码直接依赖：

```text
Python
PyTorch
Gym
D4RL
NumPy
Pyrallis
TensorBoard
TQDM
MuJoCo / mujoco-py（取决于当前 D4RL 环境）
```

可检查主要依赖：

```bash
python -c "import torch, gym, d4rl, numpy, pyrallis, tqdm; print('Environment OK')"
```

如果 D4RL 环境本身无法通过 `gym.make()` 创建，应先修复 D4RL / MuJoCo 环境，再运行本代码。

---

## 4. 最简单运行方式

只指定环境、随机种子和 GPU 即可：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

环境相关参数会根据 `env_name` 自动配置。

例如：

```bash
python warl.py --env_name="halfcheetah-medium-v2" --train_seed=0 --device="cuda:0"
```

```bash
python warl.py --env_name="walker2d-medium-replay-v2" --train_seed=10 --device="cuda:1"
```

```bash
python warl.py --env_name="antmaze-large-diverse-v2" --train_seed=20 --device="cuda:0"
```

---

## 5. 环境相关参数自动配置

当：

```text
official_preset=True
```

时，`warl.py` 会根据 `env_name` 自动设置 WSRL 官方实验对应的环境相关参数。

### AntMaze

适用于例如：

```text
antmaze-umaze-v2
antmaze-umaze-diverse-v2
antmaze-medium-play-v2
antmaze-medium-diverse-v2
antmaze-large-play-v2
antmaze-large-diverse-v2
```

自动配置：

```text
offline_algorithm = Cal-QL
actor_hidden_dims = (256, 256)
critic_hidden_dims = (256, 256, 256, 256)
num_critics = 10
online_num_min_qs = 2
state-independent policy std
online max_target_backup = True
target_backup_n_actions = 10
reward = 10 * r - 5
offline pretrain_steps = 1,000,000
CQL/Cal-QL max_target_backup = True
CQL alpha = 5.0
CQL alpha autotune = True
CQL target_action_gap = 0.8
```

### MuJoCo Locomotion

支持名称中包含：

```text
halfcheetah
hopper
walker2d
```

例如：

```text
halfcheetah-random-v2
halfcheetah-medium-v2
halfcheetah-medium-replay-v2
halfcheetah-medium-expert-v2
hopper-medium-v2
walker2d-medium-v2
```

自动配置：

```text
offline_algorithm = CQL
actor_hidden_dims = (256, 256)
critic_hidden_dims = (256, 256)
num_critics = 10
online_num_min_qs = 2
online max_target_backup = False
reward_scale = 1.0
reward_bias = 0.0
offline pretrain_steps = 250,000
CQL max_target_backup = True
CQL alpha = 5.0
```

### Kitchen

名称中包含 `kitchen` 的环境会自动识别为 Kitchen。

自动配置：

```text
offline_algorithm = Cal-QL
actor_hidden_dims = (512, 512, 512)
critic_hidden_dims = (512, 512, 512)
num_critics = 10
online_num_min_qs = 2
online max_target_backup = False
reward = r - 4
offline pretrain_steps = 250,000
CQL/Cal-QL max_target_backup = True
CQL alpha = 5.0
```

Kitchen 使用与官方配置一致的 hidden-layer 初始化设置，并记录 task completion 相关评价指标。

### Adroit-Binary

官方 preset 仅支持：

```text
pen-binary-v0
door-binary-v0
relocate-binary-v0
```

自动配置：

```text
offline_algorithm = Cal-QL
actor_hidden_dims = (512, 512)
critic_hidden_dims = (512, 512, 512)
num_critics = 10
online_num_min_qs = 2
online max_target_backup = False
reward = 10 * r + 5
offline pretrain_steps = 20,000
CQL/Cal-QL max_target_backup = True
CQL alpha = 1.0
```

Adroit-Binary 还需要官方离线数据目录。

默认查找：

```text
~/adroit_data/offpolicy_hand_data
```

也可以通过环境变量：

```bash
export DATA_DIR_PREFIX=/path/to/adroit_data
```

或者命令行显式指定：

```bash
python warl.py --env_name="pen-binary-v0" --train_seed=0 --adroit_binary_data_dir="/path/to/offpolicy_hand_data" --device="cuda:0"
```

---

## 6. 推荐的统一 1M-step 实验命令

如果论文中所有 Offline-to-Online baseline 统一使用：

```text
online steps = 1,000,000
evaluation interval = 5,000
evaluation episodes = 10
```

可以直接：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
```

不同随机种子分别运行：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
python warl.py --env_name="antmaze-umaze-v2" --train_seed=10 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
python warl.py --env_name="antmaze-umaze-v2" --train_seed=20 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
python warl.py --env_name="antmaze-umaze-v2" --train_seed=30 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
python warl.py --env_name="antmaze-umaze-v2" --train_seed=40 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
```

注意：WSRL 原论文默认 online budget 为 500k。若论文中采用统一 1M-step protocol，应在实验设置中明确说明所有方法均按照统一 online interaction budget 重新运行。

---

## 7. 需要手动控制的实验参数

环境相关算法参数默认由代码自动设置，不建议在正式 WSRL baseline 中随意修改。

实验协议参数可以通过命令行自行设置。

### Online interaction budget

```bash
--online_steps=1000000
```

默认：

```text
500000
```

### Evaluation interval

```bash
--eval_every=5000
```

默认：

```text
20000
```

### Evaluation episodes

```bash
--eval_episodes=10
```

默认：

```text
20
```

### Random seed

```bash
--train_seed=0
```

### GPU

```bash
--device="cuda:0"
```

### Log directory

```bash
--log_root="logs/WSRL_PyTorch"
```

### Checkpoint directory

```bash
--checkpoints_path="checkpoints/WSRL"
```

### Checkpoint interval

```bash
--save_every=100000
```

### TensorBoard logging interval

```bash
--log_every=5000
```

---

## 8. Offline initialization

默认：

```text
initialization_mode = pretrain
```

也就是说 `warl.py` 会先自行执行 CQL / Cal-QL offline pretraining，然后进入 WSRL online fine-tuning。

### 从 checkpoint 初始化

如果已有与本代码网络结构兼容的 PyTorch checkpoint：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --initialization_mode="load" --pretrained_checkpoint="/path/to/checkpoint.pt" --device="cuda:0"
```

默认会同时恢复：

```text
actor
critic
target critic
temperature
actor optimizer
critic optimizer
temperature optimizer
```

这与 WSRL 从 offline pretraining 继续 online fine-tuning 时保留 optimizer state 的设定一致。

---

## 9. TensorBoard

默认日志根目录：

```text
logs/WSRL_PyTorch
```

单次运行目录形式：

```text
logs/WSRL_PyTorch/<env-prefix>/<env_name>_seed<seed>_Q10_UTD4/
```

例如：

```text
logs/WSRL_PyTorch/antmaze/antmaze-umaze-v2_seed0_Q10_UTD4/
```

启动 TensorBoard：

```bash
tensorboard --logdir logs/WSRL_PyTorch --port 8088
```

---

## 10. Evaluation metrics

代码根据环境自动记录对应指标。

### AntMaze / MuJoCo locomotion

主要记录：

```text
return
episode length
d4rl_normalized_score
```

### Adroit-Binary

额外记录：

```text
success_rate
```

### Kitchen

额外记录：

```text
success_rate
num_stages_solved
```

正式论文中应优先使用与对应 benchmark 原协议一致的指标。

---

## 11. WSRL online update

warmup 完成后，每一个 environment step：

```text
1. 从 online replay buffer 随机采样 1024 transitions
2. 划分为 4 个互不重叠的 256-transition mini-batch
3. 顺序执行 4 次 critic update
4. 每次 critic update 后更新 target critic
5. 使用完整 1024-transition super-batch 执行 1 次 actor update
6. 使用完整 super-batch 执行 1 次 temperature update
```

因此：

```text
critic replay volume per environment step = 4 × 256 = 1024
actor batch size per environment step = 1024
critic updates per environment step = 4
actor updates per environment step = 1
```

---

## 12. AntMaze 特殊设置

AntMaze online WSRL 使用：

```text
10 Q-functions
randomly subsample 2 target critics
min over the selected target critics
10 candidate next actions
max target backup over candidate actions
state-independent policy std
```

即 target 构造可概括为：

```text
Q_target(s') = max_a min(Q_j1(s', a), Q_j2(s', a))
```

其中 `j1, j2` 从 10 个 critic 中随机抽取。

这与 RLPD 某些 AntMaze 配置中使用单 critic target 的设置不同，不应混淆。

---

## 13. Offline data 不参与 online fine-tuning

WSRL 的关键特征之一是：

```text
offline_data_ratio_online = 0.0
```

offline pretraining 完成后：

```text
offline dataset -> discard
online replay buffer -> initially empty
```

之后 online fine-tuning 的每一次 gradient update 都只使用在线交互得到的数据。

如果修改为在线阶段继续混合 offline data，则已经不是论文主设定下的 WSRL。

---

## 14. `official_preset`

默认：

```text
official_preset=True
```

此时环境相关算法参数会被强制设置为代码中整理的 WSRL 官方配置。

如果运行一个论文官方 preset 未覆盖的环境，会直接报错，而不是静默套用错误参数。

如果明确需要把 WSRL 机制迁移到新环境，可设置：

```bash
--official_preset=False
```

但这种情况下得到的是 **adapted WSRL configuration**，不应声称为严格的官方 WSRL benchmark reproduction。

---

## 15. 论文中的建议标注

如果使用本文件重新运行 WSRL，推荐表格中写：

```text
WSRL (PyTorch reimplementation)
```

实验设置中可以说明：

```text
We reimplemented WSRL in PyTorch following the original paper and the released JAX/Flax implementation. Environment-dependent hyperparameters follow the released configurations, while all baselines are evaluated under our unified online interaction and evaluation protocol.
```

如果修改了：

```text
online_steps
eval_every
eval_episodes
random seeds
```

这些属于统一实验 protocol，可以修改，但论文中应明确报告。

如果修改了：

```text
num_critics
online_num_min_qs
warmup_steps
UTD ratio
offline retention ratio
network architecture
reward transform
max-target backup
```

则需要重新判断是否还能作为严格的 WSRL baseline。

---

## 16. 运行前检查

建议正式批量实验前先执行：

```bash
python -m py_compile warl.py
```

然后做一个短 smoke test，例如：

```bash
python warl.py --env_name="halfcheetah-medium-v2" --train_seed=0 --pretrain_steps=100 --online_steps=100 --eval_every=50 --eval_episodes=1 --device="cuda:0"
```

注意：当 `official_preset=True` 时，环境相关的 `pretrain_steps` 会由 preset 自动设置，因此如果需要专门进行极短 smoke test，应使用独立测试配置或临时关闭 official preset；正式实验必须恢复官方 preset。

---

## 17. 注意事项

1. PyTorch 与官方 JAX/Flax 实现具有不同的随机数流、参数布局和底层计算 kernel，因此结果不应期待逐数值一致。
2. 正式比较应使用多个随机种子，并报告均值及方差/标准差。
3. D4RL、MuJoCo、Gym 版本可能影响环境行为和数据加载，正式实验应固定并记录软件版本。
4. Kitchen 应使用与 WSRL 官方实验兼容的 D4RL 环境/数据实现，否则 reward 与数据处理差异可能破坏可比性。
5. Adroit-Binary 需要单独准备对应的 offline hand dataset。
6. 本实现的核心目标是算法级对齐，而不是 JAX 与 PyTorch 的 bitwise reproduction。

---

## 18. 常用命令汇总

AntMaze：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --device="cuda:0"
```

MuJoCo：

```bash
python warl.py --env_name="walker2d-medium-v2" --train_seed=0 --device="cuda:0"
```

统一 1M-step 实验：

```bash
python warl.py --env_name="walker2d-medium-v2" --train_seed=0 --online_steps=1000000 --eval_every=5000 --eval_episodes=10 --device="cuda:0"
```

加载 offline checkpoint：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --initialization_mode="load" --pretrained_checkpoint="/path/to/checkpoint.pt" --device="cuda:0"
```

保存 checkpoint：

```bash
python warl.py --env_name="antmaze-umaze-v2" --train_seed=0 --checkpoints_path="checkpoints/WSRL" --device="cuda:0"
```

TensorBoard：

```bash
tensorboard --logdir logs/WSRL_PyTorch --port 8088
```
