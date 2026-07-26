# CFDG (Classifier-Free Diffusion Generation) - PyTorch 单文件实现

本项目提供了一个基于 PyTorch 的单文件实现，用于运行结合 IQL 底座的 CFDG (Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion Generation, ICML 2025) 数据增强框架。  

由于原论文未提供官方开源代码库，本代码是基于论文文本描述的忠实独立复现。论文中未明确指定的底层经验参数（如分类器引导权重、EMA衰减等）已作为显式的配置参数暴露，以供调整与审计。  

## ✨ 核心特性

本代码在单文件架构下完整实现了论文中规定的 CFDG 机制与缓冲区采样逻辑：

### 1. 扩散模型架构与 CFG 机制

- **网络定义**：扩散模型采用 EDM 风格的残差 MLP 去噪网络，深度为 6 层，宽度为 1024 维，使用 ReLU 激活函数。  
- **标签与联合训练**：将离线转换（transitions）与在线转换作为两个独立的分类标签。在训练时通过随机丢弃类别标签（替换为空标签）来联合训练条件与无条件去噪网络。  
- **推断引导**：采样时采用无分类器引导（Classifier-free guidance）公式：`D_cfg = (1 + w) D_cond - w D_uncond`。默认采样器采用具有 128 步的 EDM/Heun 随机采样器。  

### 2. 周期性刷新与数据生成

- **训练周期**：在线微调期间，每经过 100,000 个环境交互步，扩散模型会进行一次重新训练（Refresh）。  
- **训练细节**：每次刷新执行 100,000 步梯度更新，采用批量大小 256、Adam 优化器（学习率 3e-4）及余弦退火学习率调度。训练数据由固定的离线缓冲区与动态增长的在线缓冲区混合构成。  

### 3. 多重缓冲区与 Batch 混合比例

- **四重缓冲区**：系统维护了独立的真实离线缓冲区、真实在线缓冲区、合成离线缓冲区与合成在线缓冲区。合成缓冲区的总容量上限设定为 1,000,000。  
- **混合采样比例**：策略学习时的 Batch 严格遵循论文规定的比例构成：
	- 合成数据固定占据每个 Batch 的三分之一，从而形成真实离线、真实在线与合成数据 1:1:1 的分布。  
	- 在抽取合成数据时，在线合成数据与离线合成数据的比例严格保持在 8:2。  

### 4. 基础算法 (IQL)

- 作为策略优化底座，代码实现了标准的 IQL (Implicit Q-Learning) 算法，并在实现细节上对齐了公开的 PEX 代码库。  
- 执行 1,000,000 步的离线预训练，随后在在线阶段利用重新初始化的优化器执行 1,000,000 步的环境交互微调。内置了针对 AntMaze 任务的奖励平移预处理。  

## 🛠️ 依赖环境

- `torch`

	  

- `gym`

	  

- `d4rl`

	  

- `numpy`

	  

- `pyrallis` (配置解析)  

- `tqdm` (进度条)  

- `tensorboard` (日志记录)  

## 🚀 快速开始

代码配置通过 `pyrallis` 管理，可通过命令行直接覆盖 `TrainConfig` 中的默认超参数。

### 运行 MuJoCo 任务（默认配置）

代码默认在 `halfcheetah-medium-replay-v2` 任务上运行。  

Bash

```
python cfdg_pytorch.py --env_name halfcheetah-medium-replay-v2
```

### 运行 AntMaze 任务

AntMaze 环境通常需要更高的 advantage temperature 参数和 expectile 参数。代码中包含了自动应用此领域默认值的逻辑。  

Bash

```
python cfdg_pytorch.py --env_name antmaze-umaze-v2
```

## ⚙️ 核心超参数说明

以下为控制 CFDG 数据增强与采样逻辑的核心配置字段：

- `--env_name`: 目标环境名称（默认: `halfcheetah-medium-replay-v2`）。  
- `--diffusion_refresh_every`: 在线微调阶段扩散模型重新训练的环境步数周期（默认: `100_000`）。  
- `--diffusion_train_steps`: 每次扩散模型刷新时的网络梯度更新次数（默认: `100_000`）。  
- `--synthetic_batch_ratio`: 每个策略网络训练 Batch 中由扩散模型生成的合成数据占比（默认: `0.333`，即 1/3）。  
- `--online_synthetic_fraction`: 在抽取的合成数据中，在线合成数据所占的比重（默认: `0.8`）。  
- `--classifier_free_drop_probability`: 扩散模型训练时条件标签被丢弃（替换为空标签）的概率（默认: `0.1`）。  
- `--classifier_free_guidance_weight`: 采样时应用的无分类器引导公式中的权重 $w$（默认: `1.0`）。  
- `--diffusion_ema_decay`: 扩散模型参数的指数移动平均衰减率（默认: `0.995`）。  