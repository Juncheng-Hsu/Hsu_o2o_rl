# FamO2O (over IQL) - PyTorch 单文件实现

本项目提供了一个极简、自包含的 PyTorch 单文件实现，用于运行基于 IQL (Implicit Q-Learning) 的 **FamO2O (Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning)** 算法。  

## ✨ 核心特性

本代码在单文件架构下完整实现了 FamO2O 的核心训练机制与工程细节：

- **基础算法 (IQL)**：底层由基于 Expectile 回归的 `ValueNetwork` 和双 `QNetwork` 构成，使用优势加权回归（AWR）进行策略提取。  
- **通用策略 (Universal Policy)**：实现了条件策略网络 $\pi_u(a \vert{} s, \beta)$。标量平衡系数 $\beta$ 会经过 `sinusoidal_balance_encoding`（正弦/余弦特征编码）后与状态特征拼接，共同作为策略网络的输入。  
- **自适应平衡策略 (Balance Policy)**：实现了一个动态 Tanh-Gaussian 策略网络 $\pi_b(\beta \vert{} s)$，能够根据当前状态自适应地输出最优平衡系数，并在在线阶段利用 Q 值的梯度进行自动优化。  
- **单一经验回放池**：不再区分固定比例的 offline/online buffer。在初始化阶段将整个 D4RL 离线数据集载入单个 `ReplayBuffer` 中，在线探索得到的新数据直接追加至同个缓冲池进行全局均匀采样。  
- **环境自适应预处理**：内置了针对 D4RL 特定环境的奖励平移与缩放机制。对于 `antmaze` 任务自动应用 `rewards - 1.0`；对于 `locomotion` 任务，自动基于离线数据集的回报极小值和极大值进行归一化处理。  

## 🛠️ 依赖环境

要运行本代码，你需要安装以下 Python 库（具体版本取决于你的 CUDA/硬件环境）：

- `torch`

	  

- `gym`

	  

- `d4rl`

	  

- `numpy`

	  

- `pyrallis` (用于配置解析)  

- `tqdm` (用于进度条展示)  

- `tensorboard` (用于日志记录)  

## 🚀 快速开始

本代码支持通过 `pyrallis` 命令行参数直接覆盖 `TrainConfig` 中的默认超参数。

### 运行 AntMaze 任务（默认配置）

代码默认的超参数配置针对的是 `antmaze-umaze-v2` 任务，配置了 `expectile=0.9`，平衡系数 $\beta \in [8, 14]$ 等默认推荐参数。  

Bash

```
python FamO2O_pytorch.py --env_name antmaze-umaze-v2
```

### 运行 Locomotion 任务

对于 Mujoco 移动类任务（如 `halfcheetah`, `hopper`, `walker2d`），代码会自动应用与之匹配的推荐默认参数（`expectile=0.7`, $\beta \in [1, 5]$，平衡策略每 5 步更新一次）。  

Bash

```
python FamO2O_pytorch.py --env_name halfcheetah-medium-expert-v2
```

## ⚙️ 核心超参数说明

你可以通过命令行自由修改训练参数，以下是 `TrainConfig` 中控制 FamO2O 核心行为的参数：  

- `--pretrain_steps`: 离线预训练步数（默认: `1_000_000`）。预训练阶段 $\beta$ 会在指定的极值范围内均匀随机采样。  
- `--num_total_steps`: 在线微调总步数（默认: `1_000_000`）。在线阶段 $\beta$ 将由 Balance Policy 自适应采样生成。  
- `--expectile`: IQL 的 expectile 参数，控制价值函数的非对称性。  
- `--family_coefficient_min` / `--family_coefficient_max`: 平衡系数 $\beta$ 的允许动态范围。  
- `--balance_update_every`: 平衡策略网络更新的频率控制。  
- `--family_sin_cos_n` / `--family_sin_cos_d`: 控制正弦编码特征频率和维度的关键参数（默认 `n=10000.0`, `d=6`）。  

**日志与评估**： 训练过程中，Tensorboard 日志会自动写入到 `logs/FamO2O_PyTorch/<env_name>/<run_name>` 目录下，包含策略熵、Actor 损失、Critic 损失以及环境的实时评估结果和 D4RL 归一化得分（Normalized Score）。  