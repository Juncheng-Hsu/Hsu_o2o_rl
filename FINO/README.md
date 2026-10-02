# FINO PyTorch

`fino.py` 是 **Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning（FINO）** 的独立单文件 PyTorch 实现，包含 FQL backbone、噪声注入、多候选动作选择和 GMM 熵反馈。它不导入或依赖 `FQL/fql.py`。

算法依据：

- Shin et al., [Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning, ICLR 2026](https://arxiv.org/abs/2602.18117)。
- [作者代码](https://github.com/CTID282/FINO/tree/de49317abaf6f86e258bba7ebee991c98f2e6360)，参考提交 `de49317abaf6f86e258bba7ebee991c98f2e6360`。
- Backbone 来自 [FQL](https://github.com/seohongpark/fql)，文件内保留对应 MIT 许可说明。

**本实现以固定版本的作者代码为准。** FINO 论文与代码在噪声公式等细节上存在差异，见第 3 节。论文中可标注 **FINO (PyTorch reimplementation following the released code)**。

## 1. 运行方式

在本目录中：

```bash
python fino.py --env_name=antmaze-umaze-v2 --train_seed=0 --device=cuda:0
python fino.py --env_name=antmaze-large-diverse-v2 --train_seed=10 --device=cuda:1
python fino.py --env_name=pen-cloned-v1 --train_seed=0 --device=cuda:0
```

从仓库根目录运行时，使用 `python FINO/fino.py ...`。

检查最终配置，不加载环境或数据：

```bash
python fino.py --env_name=relocate-cloned-v1 --print_config=True
```

为了与其他算法比较相同的 100 万步在线预算，可以显式指定：

```bash
python fino.py --env_name=pen-cloned-v1 --online_steps=1000000 --replay_buffer_size=2000000
```

本文件默认保留 FINO 作者脚本的在线预算 50 万步、buffer 容量 150 万。统一实验时请同时记录交互预算、离线更新次数与 replay 容量。

## 2. 文件与本机依赖

```text
fino.py      独立单文件实现
README.md    方法、环境配置、差异与运行说明
```

核心依赖：Python 3.8+、PyTorch、NumPy、Pyrallis、TensorBoard、TQDM。D4RL 使用 Gym、D4RL 和 MuJoCo / mujoco-py。**在线 FINO 另外需要 scikit-learn**，不能省略该依赖后仍声称启用了完整的熵反馈。

2026-10-02 实测本机 WSL `o2o`：

| 包 | 版本 / 状态 |
|---|---|
| Python / PyTorch | 3.8.20 / 2.3.1+cu121 |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| Gym / D4RL | 0.23.1 / 1.1 |
| mujoco / mujoco-py | 3.2.3 / 2.1.2.14 |
| dm-control / Cython | 1.0.23 / 0.29.36 |
| Pyrallis / TensorBoard | 0.3.1 / 2.14.0 |
| scikit-learn / joblib / threadpoolctl | 未安装 |

原环境 `python -m pip check` 通过。已执行 pip 的 **dry-run 安装解析**：`scikit-learn==1.3.2` 支持 Python 3.8，满足当前 NumPy、SciPy 版本；会新增 joblib 1.4.2、threadpoolctl 3.5.0，不需要升级原有数值计算包。

用户可在 WSL 中安装：

```bash
conda activate o2o
python -m pip install "scikit-learn==1.3.2" "joblib==1.4.2" "threadpoolctl==3.5.0" "numpy==1.24.4" "scipy==1.10.1"
python -m pip check
python -c "from sklearn.mixture import GaussianMixture; import torch; print('FINO dependencies OK')"
```

本次没有修改 conda 环境或安装包。上述结果是版本依赖解析，不等于已在该环境验证安装后的 GMM 执行。程序会在加载数据和预训练前检查 scikit-learn，避免长时间训练后才因缺失包失败。

## 3. FINO 核心机制及论文/代码差异

### 3.1 噪声注入

作者代码采用：

```text
z ~ Normal(0, I), t ~ Uniform(0, 1)
x_t = (1-t)*z + t*a_data
noisy_input = x_t + noise_scale * exp(10*(t-1)) * epsilon
target_velocity = a_data - z
noise_scale = 0.1
```

即只扰动 flow matching 的网络输入，速度目标仍是 `a_data-z`。**本文件没有混入论文中另一套插值路径/方差公式。** 离线和在线更新均使用此规则，与作者代码一致。

Teacher 的 Euler 积分、student 蒸馏与 Q 最大化沿用 FQL。Bellman target 使用一步策略的一次普通采样，不使用下述多候选筛选；actor 的 Q 损失始终聚合双 Q 的均值。

### 3.2 与环境交互时的候选动作选择

```text
候选数 K = min(10, ceil(action_dim / 2))
候选来自一步策略的独立高斯潜变量采样
Q 聚合 = config.q_agg
logits = beta * Q / mean(abs(Q))
训练交互：按 softmax(logits) 随机抽取候选
评估：选择 Q 最大的候选
```

评估仍会随机生成候选，不等于完全确定性的策略。批量评估逐状态取 argmax，不使用跨 batch 的全局 argmax；单维动作也保持向量形状。

### 3.3 熵反馈与 beta

作者用当前采样策略在每个状态上生成 200 个动作，拟合 3 分量 full-covariance GMM。估计量为：

```text
estimated_entropy = H(mixture_weights) + sum_k weight_k * H(Gaussian_k)
target_entropy = -action_dim
beta = max(0, beta - 0.1 * (target_entropy - estimated_entropy))
initial_beta = 10
```

这是作者使用的 **GMM 混合熵上界近似**，不是精确混合分布熵，也不是候选 categorical 分布的熵。`beta` 是候选采样的逆温度，不是 SAC 的温度参数；不向 Bellman target 加熵奖励。

作者代码把 beta 更新放在在线阶段的评估分支中，默认每 50,000 总步更新。本实现用独立的 `entropy_update_every=50000` 保留该默认频率，触发条件为 `(completed_offline_steps + online_step) % entropy_update_every == 0`，避免仅修改 `eval_every` 就改变算法。每次使用最近一次训练 batch 的全部状态，评估后更新 beta。

数值与复现处理：Q 归一化分母加极小下界防止 0/0；GMM 显式设置随机种子；动作采样按块计算控制内存。PyTorch 与 JAX 的随机数流不同，不保证逐位一致。

## 4. 自动配置

本次支持 **33 个 D4RL 状态任务**，不包含 OGBench、像素任务或 Adroit binary：

| 类别 | 任务 | 数据质量 / 版本 | 数量 |
|---|---|---|---:|
| MuJoCo locomotion | halfcheetah、hopper、walker2d | random、medium、medium-replay、medium-expert、expert；v2 | 15 |
| AntMaze | umaze、umaze-diverse、medium-play、medium-diverse、large-play、large-diverse | v2 | 6 |
| Adroit | pen、door、hammer、relocate | human、cloned、expert；v1 | 12 |

完整环境名示例：`walker2d-expert-v2`、`antmaze-medium-play-v2`、`relocate-human-v1`。统一环境与数据集不改变 FINO 的网络、采样比例和更新规则；实验预算可显式传参统一。

作者提供的 D4RL O2O 配置：

| 环境 | alpha | q_agg | discount |
|---|---:|---|---:|
| antmaze-umaze-v2 / antmaze-umaze-diverse-v2 | 10 | mean | 0.99 |
| antmaze-medium-play-v2 / antmaze-medium-diverse-v2 | 10 | mean | 0.99 |
| antmaze-large-play-v2 / antmaze-large-diverse-v2 | 3 | mean | 0.99 |
| pen-cloned-v1 / door-cloned-v1 / hammer-cloned-v1 | 1000 | min | 0.99 |
| relocate-cloned-v1 | 10000 | **mean** | 0.99 |

`relocate-cloned-v1` 保留作者 FINO README 命令中的 `mean`，不照搬 FQL 的 `min`。配置均默认 `normalize_q_loss=False`。

作者未报告的任务使用已确认的固定迁移规则：

| 环境 | alpha | q_agg | Q loss 归一化 | 参数来源 |
|---|---:|---|---|---|
| pen / door / hammer 的 human、expert | 1000 | min | False | 沿用同任务 cloned 参数 |
| relocate 的 human、expert | 10000 | mean | False | 沿用 relocate-cloned 参数 |
| 全部 15 个 locomotion 任务 | 1 | mean | True | 固定迁移规则，各数据质量相同 |

以上 `discount=0.99`、离线更新默认 100 万次，日志与 `config.json` 中的 `config_source` 标记为 `adapted_*`。这些配置未声称经过调优；自动配置不等于自动搜索超参数。`online_steps=0` 仅跳过在线阶段，保留 FINO 的离线预训练配置，不切换成 FQL 的纯离线参数表。

`alpha`、`discount`、`q_agg`、`normalize_q_loss`、`offline_steps` 使用 `None` 表示自动补全。用户显式传参优先并记录在 `config_source` 中。`auto_env_config=False` 需要显式指定 `alpha`。`strict_official_only=True` 拒绝没有作者任务配置的迁移环境，不代表强制用户的所有预算和网络覆盖值都与作者相同。

## 5. 默认训练设置

| 项目 | 默认值 |
|---|---|
| Actor / critic 隐藏层 | `(512, 512, 512, 512)` |
| Critic / actor LayerNorm | True / False |
| 激活 / 初始化 | 近似 GELU / Xavier uniform，零偏置 |
| LayerNorm 位置 / epsilon | 激活之后 / 1e-6 |
| Learning rate / batch size | 3e-4 / 256 |
| Target tau / Euler steps | 0.005 / 10 |
| Offline updates / online interactions | 1,000,000 / 500,000 |
| UTD / replay capacity | 1 / 1,500,000 |
| Noise scale / initial beta | 0.1 / 10 |
| Entropy interval / samples / components | 50,000 / 200 / 3 |

默认统一 buffer 预填充离线数据，随后插入在线数据并均匀采样；`balanced_sampling=True` 可切换为独立在线 buffer 的 50/50 混合。无随机动作 warmup、无额外执行动作噪声。

Critic、flow teacher、一步 student 的损失在更新前参数上计算；单个 Adam 更新三个网络。Teacher 蒸馏目标停止梯度；actor Q 梯度不更新 critic。目标网络沿用作者代码的更新前 critic 做 Polyak 平均。

AntMaze 训练奖励 `r-1`，其他 D4RL 任务使用原始奖励。评估始终使用原始回报。时间截断仍 bootstrap，真正终止不 bootstrap。输入限定为向量状态和 `[-1,1]` 连续动作。

## 6. 日志、checkpoint 与验证

```bash
python fino.py --env_name=antmaze-umaze-v2 --train_seed=0 \
  --log_root=logs/FINO --checkpoints_path=checkpoints/FINO
tensorboard --logdir logs/FINO
```

运行目录包含最终配置 `config.json`、TensorBoard events，以及可选 checkpoint。主要 tags 为 `offline/*`、`offline_evaluation/*`、`online/*`、`evaluation/*`、`exploration/*`，另有 `online_sampling/entropy_estimate` 和 `online_sampling/beta`。

在线评估横轴为真实累计交互次数，0 表示离线初始化；最后一步会补评估。`eval_every=0` 禁用评估，但不会禁用 FINO 熵反馈。

`offline_final.pt` 可通过 `--pretrained_checkpoint=/absolute/path/to/offline_final.pt` 加载并直接进入在线阶段；检查网络与关键算法参数，并恢复 Adam、beta、计数和随机流。只接受本文件生成的离线终点，不转换 JAX checkpoint。在线 checkpoint 不含 replay 和环境状态，不提供精确中断续训。

验证包括小网络合成 batch 的更新、噪声公式、梯度隔离、候选采样、Bellman mask、GMM 熵与 beta、checkpoint，以及合成环境的完整训练/评估循环。核心检查在 WSL `o2o` 中通过；真实 GMM 检查使用本机另一个已安装 scikit-learn 的 Python 环境。未运行真实 D4RL 训练，不宣称已复现论文分数。
