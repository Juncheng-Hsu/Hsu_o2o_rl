# RLPD (Reinforcement Learning from Prior Data) - PyTorch 单文件实现

本项目提供了一个基于 PyTorch 的单文件状态空间 RLPD 算法实现。本代码旨在保持单文件风格的极简性与易读性，同时在工程细节上高度还原官方 JAX/Flax 实现，但在核心调度上严格遵循原论文的伪代码设计。  

## ✨ 核心特性与工程对齐

本代码在底层工程细节上最大程度地抹平了 PyTorch 与 JAX/Flax 的框架差异，还原了官方实验的细节：

- **数据采样 (50/50 Interleaving)**：严格实现了 50% 离线专家数据与 50% 在线探索数据的交错混合采样，保证每个 mini-batch 内的数据分布绝对均匀。  
- **网络架构与目标计算**：完整实现了支持任意数量的 Critic 集成网络（Ensemble Critic），支持通过 `num_min_qs` 参数控制目标 Q 值的计算方式。  
- **框架差异对齐**：
	- 手动将 `LayerNorm` 的 `eps` 对齐为 Flax 默认的 `1e-6`。  
	- 在构建 Critic 优化器时，精准排除了对 Bias（偏置项）的权重衰减（Weight Decay），对齐 Optax 的默认行为。  
- **数值稳定性**：采用了数值更稳定的雅可比行列式计算公式来处理 Tanh 动作空间的对数概率（Log Prob）。  
- **D4RL 环境预处理**：内置了官方代码库中针对 `antmaze` 系列任务的标准预处理逻辑（即 `rewards - 1.0` 的奖励平移）。  

## ⚠️ 关键声明：关于 UTD 调度机制 (与官方 JAX 代码的区别)

在 UTD (Update-To-Data) 更新调度机制上，本代码**故意偏离了官方释出的 JAX 代码库，而是选择严格遵循 RLPD 原论文中的 Algorithm 1 伪代码**：  

- **本代码的非对称调度 (Algorithm 1)**：在每一个环境交互步后，一个包含 `batch_size * utd_ratio` 样本的 super-batch 会被拆分为 `utd_ratio` 个 mini-batches。Critic 和 Target Critic 会在每个 mini-batch 上连续更新（共更新 `utd_ratio` 次）；**但在所有 Critic 更新结束后，Actor 和 Temperature 仅使用最终的一个 mini-batch 恰好更新 1 次**。  
- **官方 JAX 仓库的做法**：在官方代码库中，Critic、Actor 和 Temperature 在每一次 UTD 迭代内部都会被同步更新（即 Actor 与 Alpha 也会伴随更新 `utd_ratio` 次）。  

**设计选择说明**：保留单次低频的 Actor 更新可以避免策略在 Q 值尚未稳定收敛时发生过拟合或崩溃，这一非对称更新调度忠实还原了使用者要求的原论文算法逻辑。  

## 🚀 默认配置与快速开始

代码默认采用论文中针对 `antmaze` 任务的参考超参数配置：  

- **环境**: `antmaze-umaze-v2`

	  

- **Critics 数量**: 10  

- **UTD Ratio**: 20  

- **网络结构**: 3 层 256 维度隐藏层  

- **随机探索步数**: 5,000 步  

- **在线训练步数**: 300,000 步  

### 运行环境与启动命令

本实现为一个独立且自包含的 Python 单文件（`rlpd_paper_pytorch.py`）。确保你的环境中安装了 `torch`, `gym` 以及 `d4rl` 后，即可直接运行：  

Bash

```
python rlpd_paper_pytorch.py --env_name antmaze-umaze-v2
```