# OPT (Online Pre-Training for Offline-to-Online RL) - PyTorch 单文件实现

本项目提供了一个基于 PyTorch 的单文件实现，用于运行 **OPT (Online Pre-Training for Offline-to-Online Reinforcement Learning, ICML 2025)** 算法。  

本代码对官方释出的 PyTorch 仓库进行了深度的结构重组，将其整合为易读的单文件风格，同时完美保留了官方的算法时间线与元学习级别的控制流。  

## ✨ 核心特性与工程对齐

本代码在单文件架构下完整实现了 OPT 论文中的四大关键阶段及所有核心机制：

- **双基线支持**：内置支持 **TD3+BC**（用于 MuJoCo 等 Locomotion 任务）和 **SPOT**（包含 VAE，用于 AntMaze/Adroit 等具有隐式动作支持集约束的任务）两种底层离线算法。  
- **在线预训练 (Online Pre-training)**：
	- 通过自定义的 `HotPlug` 模块实现了可微的内部步更新（Differentiable Inner Update）。  
	- 利用 `create_graph=True` 精准实现了对内部梯度更新链条的高阶求导，并根据公式应用了基于损失比例的动态梯度加权（Gradient Weighting）。  
- **在线微调的动态 Q 值融合**：在最终的微调阶段，Actor 网络通过最大化动态融合的 Q 值进行更新：`blended_q = (1.0 - kappa) * old_q + kappa * new_q`，其中 `kappa` 采用线性退火调度。  
- **密度比优先回放池 (Density-Ratio Prioritized Replay)**：
	- 实现了一个独立的 `DensityRatio` 判别器网络，用于评估数据分布。  
	- 基于 `SumTree` 数据结构构建了优先采样回放池，利用估算的密度比为在线微调样本分配采样优先级。  

## 🛠️ 依赖环境

- `torch`

	  

- `gym`

	  

- `d4rl`

	  

- `numpy`

	  

- `pyrallis` (用于配置解析)  

- `tqdm` (用于进度条展示)  

- `tensorboard` (用于日志记录)  

## 🚀 快速开始

本代码支持通过 `pyrallis` 命令行参数覆盖 `TrainConfig` 中的超参数。同时内置了原论文中三个代表性的官方配置预设（Presets）。  

### 1. 运行 MuJoCo 任务 (TD3+BC + OPT)

预设环境：`hopper-medium-v2`

  

Bash

```
python OPT_pytorch_2.py --official_preset hopper-medium-v2
```

预设环境：`walker2d-random-v2`

  

Bash

```
python OPT_pytorch_2.py --official_preset walker2d-random-v2
```

### 2. 运行 AntMaze 任务 (SPOT + OPT)

预设环境：`antmaze-umaze-v2`

  

Bash

```
python OPT_pytorch_2.py --official_preset antmaze-umaze-v2
```

(注意：使用 `official_preset` 将自动覆盖算法类型、学习率、奖励归一化设置及环境名称等参数以完全对齐官方实验配置。)  

## ⚙️ 核心参数说明

如果不使用预设，或者需要微调特定任务的实验参数，可以参考以下 `TrainConfig` 中的核心字段：  

### 时间线控制参数

- `--offline_steps`: 离线预训练的梯度更新步数（默认: `1_000_000`）。  
- `--n_tau`: 冻结离线策略收集初始在线数据的步数（默认: `25_000`）。满足该步数后触发在线预训练阶段。  
- `--online_pretrain_steps`: 在线预训练阶段的内部/外部循环梯度更新总次数（默认: `50_000`）。  
- `--online_steps`: 在线微调的交互总步数（默认: `1_000_000`）。  

### OPT 特有机制参数

- `--online_pretrain_inner_lr`: 在线预训练内部循环中，替代更新 `HotPlug` 参数的学习率（默认: `3e-4`）。  
- `--priority_temperature`: 用于密度比转优先级的温度缩放系数（默认: `5.0`）。  
- `--kappa`: 在线微调初始时刻的 Q 值融合权重。  
- `--kappa_end`: 退火结束时的目标融合权重。  
- `--kappa_cool_steps`: `kappa` 从初始值线性退火至 `kappa_end` 所需的步数。  