# APL (Adaptive Policy Learning) - PyTorch 单文件实现

本项目提供了一个独立、极简的 PyTorch 单文件实现，用于运行 **Adaptive Policy Learning for Offline-to-Online Reinforcement Learning (AAAI 2023)** 框架下定义的两种核心算法变体：**GCQL** 与 **GCTD3BC**。

由于原论文未提供官方开源代码库，本实现是基于论文 Algorithm 1 伪代码及其实验细节进行的学术级忠实复现。

---

## ✨ 核心算法机制

本代码在单个文件中实现了 APL 框架的三个核心组件：

### 1. 在线-离线重放缓冲区 (OORB)

* **双层结构**：实现了一个容量为 3,000,000 的 `offline_buffer`（包含完整 D4RL 离线数据及所有后续收集的在线数据），以及一个容量为 20,000 的 FIFO `online_buffer`（仅保存最近的在线近同策略数据）。


* **动态采样门控**：在交互步数达到阈值 `T_s` 后，每次梯度更新以概率 `p` 从 `online_buffer` 中采样整个 batch，否则从 `offline_buffer` 中采样。采样的来源直接决定了后续保守性惩罚的门控开关（即论文中的自适应权重 $W(s,a)$）。



### 2. GCQL (Greedy-Conservative Q-ensemble Learning)

* 在线更新底座为 REDQ/SAC 框架。


* **源门控 CQL 惩罚**：CQL 保守性惩罚仅对来自 `offline_buffer` 的 batch 生效；对于来自 `online_buffer` 的 batch，关闭惩罚以执行贪婪的乐观更新。


* 默认采用 5 个 Critic 且通过 `num_min_qs=2` 进行最小 Q 值计算。



### 3. GCTD3BC (Greedy-Conservative TD3+BC)

* 在线更新底座为 TD3 框架，并默认对观测状态进行归一化处理。


* **源门控 BC 惩罚**：行为克隆（Behavior Cloning）损失仅约束来自 `offline_buffer` 的 batch；对 `online_buffer` 的 batch 仅执行标准 TD3 的策略最大化更新。



---

## 📅 块交错训练调度 (Block-Interleaved Schedule)

代码的主控制流严格遵循了原论文实验章节的块交错设计，以避免在线与离线梯度的频繁冲突：

1. **初始离线预训练**：执行 100,000 步的纯离线梯度更新。


2. **在线微调迭代块**：
* 连续收集 1,000 步在线交互数据，并同时存入两个缓冲区。


* 冻结环境交互，执行 10,000 步连续的网络梯度更新（基于 OORB 概率采样）。


* 重复此循环，直至在线环境交互达到总设定的 100,000 步。





---

## 🛠️ 依赖环境

执行本代码需要安装以下核心依赖：

* `torch`

* `gym`

* `d4rl`

* `numpy`

* `pyrallis`

* `tqdm`

* `tensorboard`


---

## 🚀 快速开始

代码使用 `pyrallis` 进行配置管理。你可以通过命令行参数快速启动不同的 APL 变体。

### 运行 GCQL 算法（默认配置）

默认配置会在 `hopper-medium-replay-v2` 任务上运行 GCQL，采用论文推荐的 `p=0.5` 采样概率。

```bash
python APL_pytorch.py --algorithm GCQL --env_name hopper-medium-replay-v2

```

### 运行 GCTD3BC 算法

在启动 GCTD3BC 时，需要将算法参数变更为 `GCTD3BC`。代码会自动将其默认采样概率解析为论文推荐的 `p=0.1`，并开启状态归一化。

```bash
python APL_pytorch.py --algorithm GCTD3BC --env_name hopper-medium-replay-v2

```

---

## ⚙️ 核心超参数说明

以下是控制 OORB 与 APL 算法行为的关键参数，可按需通过命令行覆盖：

* `--algorithm`: 选择使用的变体，可选值为 `GCQL` 或 `GCTD3BC`。


* `--online_sampling_probability`: 从在线 FIFO 缓冲区采样的概率 $p$。若不指定，GCQL 默认为 0.5，GCTD3BC 默认为 0.1。


* `--online_sampling_start`: 开始允许从在线池采样的初始在线交互步数阈值 $T_s$（默认: `10_000`）。


* `--initial_offline_updates`: 在正式开始在线收集前的纯离线预训练更新步数（默认: `100_000`）。


* `--interaction_steps_per_iteration`: 每个微调迭代块中的在线数据收集步数 $T_{on}$（默认: `1_000`）。


* `--updates_per_iteration`: 每个微调迭代块中的网络梯度更新次数 $T_{off}$（默认: `10_000`）。


* `--online_buffer_size`: 在线 FIFO 缓冲区的容量（默认: `20_000`）。


* `--offline_buffer_size`: 离线/增长缓冲区的容量（默认: `3_000_000`）。