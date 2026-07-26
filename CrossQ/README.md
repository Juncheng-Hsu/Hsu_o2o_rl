# CrossQ - PyTorch 单文件实现

本项目提供了一个极简、自包含的 PyTorch 单文件实现，用于运行 **CrossQ (ICLR 2024)** 算法。  

官方原始代码基于 JAX/Flax 框架开发，本文件在 PyTorch 框架下对官方仓库进行了算法与工程层面的等效重构，完整保留了区分 CrossQ 与传统 SAC 的核心架构革新。  

## ✨ 核心机制与架构革新

在单文件架构下，本代码严格实现了 CrossQ 的六项关键突破设计：  

- **彻底移除目标网络 (No Target Critic)**：CrossQ 废除了传统 Off-Policy 强化学习中的目标网络（Target Net），在计算自举 TD 目标时，直接使用当前的 Critic 网络进行评估，大幅减少了内存开销与滞后偏差。  
- **联合前向传播 (Joint Forward Pass)**：为确保当前状态 $(s, a)$ 与下一状态 $(s', a')$ 在批量归一化时的分布统计量完全同步，代码将当前与下阶段数据在批次维度拼接为 `2B` 大小，在 Critic 中执行单次联合前向计算。  
- **批量重归一化 (Batch Renormalization, BRN)**：
	- 在 Actor 和 Critic 的所有隐藏层及输入层嵌入 `BatchRenorm1d` 模块。  
	- 内置 100,000 步的预热期（Warm-up），预热期内直接使用当前微批次统计量；达到阈值后自动启动基于 `r_max = 3.0` 和 `d_max = 5.0` 的滑动平均截断校正。  
- **非对称网络容量与初始化**：Critic 宽度扩大至 2048 单元（`2048 x 2048`），Actor 保持 256 单元（`256 x 256`）；所有线性网络层严格采用 LeCun Normal 正交初始化。  
- **非对称延迟更新调度**：采用 `UTD = 1` 且策略延迟 `policy_delay = 3` 的调度策略，即 Critic 网络每步都进行梯度更新，而 Actor 网络和熵温度参数 $\alpha$ 每 3 步更新一次。  
- **Adam 优化器动量适配**：为在大容量网络下稳定训练，Actor 和 Critic 优化器的 `beta_1` 均适配调整为 0.5，温度优化器保持 0.9。  

## 🛠️ 依赖环境

执行本代码需要安装以下基础深度学习与强化学习组件：  

- `torch`

	  

- `gymnasium` (或兼容旧版本的 `gym`)  

- `numpy`

	  

- `pyrallis` (用于命令行参数解析)  

- `tqdm` (用于进度条展示)  

- `tensorboard` (用于训练日志记录)  

## 🚀 快速开始

代码通过 `pyrallis` 管理所有超参数。你可以直接在命令行中指定参数来启动不同的控制任务。  

### 1. 运行 MuJoCo 默认基准任务

代码默认在 `Humanoid-v4` 任务上运行，该任务是原论文及官方 README 中展示 CrossQ 大容量学习能力的标准环境：  

Bash

```
python crossq_pytorch.py --env_name Humanoid-v4
```

### 2. 运行其他 MuJoCo / DM Control 任务

你可以随时切换环境名称，代码内部内置了针对不同环境（如 `Swimmer-v4` 或 `dm_control` 系列）的总总步数与折扣因子（`discount`）自动适配逻辑：  

Bash

```
# 运行 HalfCheetah 任务
python crossq_pytorch.py --env_name HalfCheetah-v4

# 运行 Swimmer 任务 (代码会自动将 discount 调整为官方指定的 0.9999)
python crossq_pytorch.py --env_name Swimmer-v4
```

## ⚙️ 核心参数对照表

通过命令行可以随时覆盖 `TrainConfig` 中的默认控制参数。以下为影响 CrossQ 核心表现的关键超参数：  

| **参数名称**           | **默认值**     | **参数说明**                                |
| ---------------------- | -------------- | ------------------------------------------- |
| `--env_name`           | `Humanoid-v4`  | 训练的目标环境名称                          |
| `--actor_hidden_dims`  | `(256, 256)`   | Actor 策略网络隐藏层维度                    |
| `--critic_hidden_dims` | `(2048, 2048)` | Critic 价值网络隐藏层维度                   |
| `--batch_size`         | `256`          | 每次梯度更新的批次大小                      |
| `--policy_delay`       | `3`            | Actor 与温度参数延迟于 Critic 更新的频次    |
| `--brn_warmup_steps`   | `100000`       | Batch Renorm 不开启截断校正的预热步数       |
| `--brn_momentum`       | `0.99`         | BRN 运行均值/方差的动量参数 (Flax 命名规范) |
| `--actor_adam_beta1`   | `0.5`          | Actor Adam 优化器的一阶动量衰减系数         |
| `--critic_adam_beta1`  | `0.5`          | Critic Adam 优化器的一阶动量衰减系数        |
| `--learning_starts`    | `5000`         | 开始网络梯度更新前的随机探索交互步数        |

## ⚠️ 框架差异与数值对齐说明

本项目处于 PyTorch 框架下，而原作者释出的基线代码为 JAX/Flax 框架。由于两大框架在**底层 CUDA 算子实现 (Low-level Kernels)**、**参数内存布局 (Parameter Layouts)** 以及 **伪随机数生成流 (PRNG Streams)** 上存在本质的非兼容性，使用相同随机种子运行无法得到逐比特（Bitwise）一样的轨迹与参数。  

本代码保证了在**数学控制流、更新时序、损失函数构建以及归一化逻辑**上的 1:1 算法等效性，可作为高度可信的 PyTorch 基线用于对比与扩展研究。  