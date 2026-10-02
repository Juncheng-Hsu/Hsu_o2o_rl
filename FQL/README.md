# FQL PyTorch

`fql.py` 是 **Flow Q-Learning（FQL）** 的独立单文件 PyTorch 实现，用于状态输入的 offline RL 和 offline-to-online RL。网络、数据加载、replay buffer、训练、评估和 checkpoint 均在该文件中，不依赖本仓库其他算法文件。

算法依据：

- Park et al., [Flow Q-Learning, ICML 2025](https://proceedings.mlr.press/v267/park25f.html)。
- [作者代码](https://github.com/seohongpark/fql/tree/e8cd16eb490332924dfa2492097219f181765933)，参考提交 `e8cd16eb490332924dfa2492097219f181765933`。
- 网络与损失对应 `agents/fql.py`、`utils/networks.py`；环境参数对应作者 README 的 **Offline-to-online RL** 与 **Offline RL** 两个不同部分。

论文中可标注 **FQL (PyTorch reimplementation)**。这不是作者 JAX 代码的逐位数值复现，也未通过完整训练验证原论文分数。

## 1. 运行方式

在本目录中：

```bash
python fql.py --env_name=antmaze-umaze-v2 --train_seed=0 --device=cuda:0
python fql.py --env_name=antmaze-large-diverse-v2 --train_seed=10 --device=cuda:1
python fql.py --env_name=pen-cloned-v1 --train_seed=0 --device=cuda:0
```

从仓库根目录运行时，使用 `python FQL/fql.py ...`。

先检查自动配置，不创建环境、不加载数据、不训练：

```bash
python fql.py --env_name=pen-cloned-v1 --print_config=True
```

纯离线训练：

```bash
python fql.py --env_name=pen-cloned-v1 --online_steps=0
```

**注意：纯离线与 O2O 的作者参数不同。** 例如 `pen-cloned-v1` 的 O2O 配置是 `alpha=1000`、离线训练 100 万步；纯离线配置是 `alpha=10000`、训练 50 万步。自动配置会根据是否存在在线阶段选择对应参数，不能混用两张参数表。

## 2. 文件与依赖

```text
fql.py       独立单文件实现
README.md    方法、环境配置与运行说明
```

核心依赖：Python 3.8+、PyTorch、NumPy、Pyrallis、TensorBoard、TQDM。D4RL 任务另外需要 Gym、D4RL 和相应的 MuJoCo / mujoco-py 环境。

环境相关导入延迟到创建环境时，`--help`、`--print_config=True` 和算法单元检查不会触发 D4RL 或 MuJoCo 的加载。

### 本机 WSL `o2o` 兼容性检查

2026-10-02 检查到：

| 包 | 已安装版本 |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.3.1+cu121 |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| Gym / D4RL | 0.23.1 / 1.1 |
| mujoco / mujoco-py | 3.2.3 / 2.1.2.14 |
| dm-control / Cython | 1.0.23 / 0.29.36 |
| Pyrallis / TensorBoard | 0.3.1 / 2.14.0 |

`python -m pip check` 通过。FQL 核心更新已在该环境中进行 CPU 小规模检查。运行 D4RL 不需要为本实现安装 JAX、Flax、Gymnasium 或 scikit-learn。

## 3. 核心机制

每次更新在同一组更新前参数上计算三个部分：

1. **Critic：**用当前一步策略采样下一动作，目标双 Q 取 `mean` 或 `min`，构造不带熵项的 Bellman target。
2. **Flow matching：**采样高斯起点 `z` 和时间 `t`，对 `(1-t)z + t*a_data` 回归速度 `a_data-z`。
3. **一步策略：**用同一份噪声生成 teacher 与 student 动作，以蒸馏 MSE 约束 student，同时最大化当前双 Q 的均值。

总损失：

```text
critic_loss + flow_matching_loss + alpha * distillation_loss - mean(Q)
```

`normalize_q_loss=True` 时，只对最后的 Q 优化项除以停止梯度后的 `mean(abs(Q))`。它不是 reward normalization。

实现中的重要细节：

- Teacher 用 10 步 Euler 积分，只裁剪最终输出；teacher 蒸馏目标停止梯度。
- 蒸馏比较 student 的未裁剪输出，Q 查询使用裁剪至 `[-1,1]` 的输出。
- Actor 的 Q 梯度传到动作和 student，不传到 critic 参数。
- `q_agg` 只控制 Bellman target 的双 Q 聚合；actor 始终使用双 Q 的均值。
- MLP 使用 Xavier uniform、零偏置、近似 GELU；critic 是 **Linear → GELU → LayerNorm**，不是 LayerNorm → GELU。
- 单个 Adam 优化 critic、teacher、student；actor 使用更新前 critic。
- 作者代码的 target update 读取更新前 critic，本文件保留该顺序。
- 评估也保留潜变量高斯采样；将噪声置零并不等于作者的 FQL 评估策略。

## 4. 自动配置与覆盖规则

本次支持 **33 个 D4RL 状态任务**，不包含 OGBench、像素任务或 Adroit binary：

| 类别 | 任务 | 数据质量 / 版本 | 数量 |
|---|---|---|---:|
| MuJoCo locomotion | halfcheetah、hopper、walker2d | random、medium、medium-replay、medium-expert、expert；v2 | 15 |
| AntMaze | umaze、umaze-diverse、medium-play、medium-diverse、large-play、large-diverse | v2 | 6 |
| Adroit | pen、door、hammer、relocate | human、cloned、expert；v1 | 12 |

完整环境名示例：`halfcheetah-expert-v2`、`antmaze-large-diverse-v2`、`hammer-human-v1`。统一的是环境和数据集，算法的网络、采样比例及更新规则仍按 FQL 实现；实验预算可以显式传参统一。

`alpha`、`discount`、`q_agg`、`normalize_q_loss`、`offline_steps` 默认是 `None`，根据环境补全。显式命令行值优先，例如：

```bash
python fql.py --env_name=antmaze-large-play-v2 --alpha=5 --offline_steps=500000
```

这会保留 `alpha=5`，并在 `config_source` 中记录参数覆盖。

`auto_env_config=False` 时必须显式给出 `alpha`，其他未填写值采用通用默认值。`strict_official_only=True` 用于拒绝没有作者任务参数依据的迁移环境；它不强制所有训练预算、网络和用户覆盖参数完全等于作者值。

作者提供的 D4RL O2O 配置：

| 环境 | alpha | target Q 聚合 | discount |
|---|---:|---|---:|
| antmaze-umaze-v2 / antmaze-umaze-diverse-v2 | 10 | mean | 0.99 |
| antmaze-medium-play-v2 / antmaze-medium-diverse-v2 | 10 | mean | 0.99 |
| antmaze-large-play-v2 / antmaze-large-diverse-v2 | 3 | mean | 0.99 |
| pen-cloned-v1 / door-cloned-v1 / hammer-cloned-v1 | 1000 | min | 0.99 |
| relocate-cloned-v1 | 10000 | min | 0.99 |

这些配置的 `normalize_q_loss=False`，默认离线 100 万次更新。D4RL AntMaze 的离线和在线训练奖励均为 `r-1`；Adroit 使用原始奖励。评估使用原始环境回报和 D4RL normalized score × 100。

作者未报告的 O2O 任务按已确认的固定规则迁移：

| 环境 | alpha | target Q 聚合 | Q loss 归一化 | 参数来源 |
|---|---:|---|---|---|
| pen / door / hammer 的 human、expert | 1000 | min | False | 沿用同任务 cloned O2O 参数 |
| relocate 的 human、expert | 10000 | min | False | 沿用 relocate-cloned O2O 参数 |
| 全部 15 个 locomotion 任务 | 1 | mean | True | 固定迁移规则，各数据质量相同 |

以上 `discount=0.99`、离线更新默认 100 万次；locomotion 使用原始奖励，不进行奖励或状态标准化。日志与 `config.json` 的 `config_source` 标记为 `adapted_*`。这是自动选择固定配置，不是自动调参，也不表示作者在这些任务上验证过该配置。

当 `online_steps=0` 时，Adroit 改用作者纯离线参数：

| 任务 | human alpha | cloned alpha | expert alpha |
|---|---:|---:|---:|
| pen | 10000 | 10000 | 3000 |
| door | 30000 | 30000 | 30000 |
| hammer | 30000 | 10000 | 30000 |
| relocate | 10000 | 30000 | 30000 |

纯离线 Adroit 仍用 `q_agg=min`；AntMaze 的 alpha 与上表相同。两类任务的纯离线更新默认 50 万次。

## 5. 训练协议

| 项目 | 默认值 |
|---|---|
| Actor / critic 隐藏层 | `(512, 512, 512, 512)` |
| Critic / actor LayerNorm | True / False |
| LayerNorm epsilon | 1e-6 |
| Learning rate / batch size | 3e-4 / 256 |
| Target tau / flow steps | 0.005 / 10 |
| Online interactions / UTD | 1,000,000 / 1 |
| Replay capacity | 2,000,000，至少容纳全部初始数据再加一条在线数据 |
| Online sampling | 单一 buffer，离线数据预填充后均匀采样 |

默认没有随机动作 warmup、额外动作噪声或 SAC 熵奖励。每次在线交互后更新一次三个网络。`balanced_sampling=True` 是作者代码提供的可选设置，使用独立的在线 buffer 并按 50/50 混合，不是默认 FQL 协议。

所有训练预算与日志控制可由用户指定。评估频率默认每 5000 步、10 个 episode，与作者脚本的日志协议不同；不改变 FQL 学习规则。在线日志横轴从 0 开始记录预训练后评估，此后的步数是真实累计交互次数，并在最终交互步补评估（`eval_every=0` 可禁用）。

时间截断仍 bootstrap；真正终止不 bootstrap。只接受向量观测和边界为 `[-1,1]` 的连续动作，不静默重缩放其他动作空间。

## 6. 日志、checkpoint 与可重复性

```bash
python fql.py --env_name=antmaze-umaze-v2 --train_seed=0 \
  --log_root=logs/FQL --checkpoints_path=checkpoints/FQL
tensorboard --logdir logs/FQL
```

每次运行创建独立目录，保存最终解析的 `config.json` 和 TensorBoard events。主要 tags：

```text
offline/*                 离线训练损失
offline_evaluation/*      离线评估，横轴为更新次数
online/*                  在线训练损失与累计更新次数
evaluation/*              在线评估，横轴为环境交互次数
exploration/*             在线交互 episode 回报和长度
```

Checkpoint 包含网络、Adam 状态、算法随机流状态、配置与训练计数。`offline_final.pt` 可以用于跳过离线预训练直接进入在线阶段：

```bash
python fql.py --env_name=antmaze-umaze-v2 \
  --pretrained_checkpoint=/absolute/path/to/offline_final.pt
```

加载时检查环境、网络结构和关键算法参数。纯离线配置保存的 checkpoint 若与 O2O 参数不同，会明确拒绝，需显式给出一致参数。只支持本文件的 PyTorch checkpoint；不转换 JAX checkpoint。在线 replay 与环境状态没有保存，因此在线 checkpoint 用于保存结果，不提供精确中断续训。

训练、交互、replay、评估使用明确的随机种子；评估不消耗训练随机流。不同框架、设备与底层算子仍可能产生数值差异。

## 7. 验证范围

已完成小网络合成 batch 检查：梯度隔离、Bellman mask、Euler 积分、目标网络更新顺序、buffer 回绕、checkpoint 恢复，以及使用合成环境的离线到在线循环和最终步 TensorBoard 记录。

这些检查不下载 D4RL 数据、不运行真实环境训练，也不证明论文性能已复现。
