# Cal-QL - PyTorch 单文件实现

本项目提供 Cal-QL (Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning, NeurIPS 2023) 算法的 PyTorch 单文件实现。  

该代码移植自原作者发布的 JAX/Flax 官方代码库，在算法逻辑与底层约束机制上与官方释出版本保持一致。  

## 核心算法机制

本实现基于 SAC+CQL 架构，并包含了 Cal-QL 论文中定义的核心校准机制与在线微调调度：  

- **Q 值校准 (Calibration)**：在计算 CQL 的保守惩罚项 (log-sum-exp) 之前，仅对由策略网络采样得到的动作的 Q 值应用下界约束（下界为参考策略的蒙特卡洛回报 `mc_returns`）。离线数据动作和随机动作的 Q 值不进行截断。  
- **SAC 目标备份 (Target Backup)**：支持 Max-Q 目标备份计算，即在多个（默认 10 个）下一状态策略动作中取最小 Q 值的最大值作为目标。针对官方发布的稀疏任务，默认关闭熵备份 (`backup_entropy=False`)。  
- **拉格朗日乘子机制**：完整支持 CQL 的拉格朗日权重自适应调节，允许设定目标动作间隙 (`cql_target_action_gap`)。  
- **轨迹级在线收集与混合重放**：
	- 在线阶段通过收集完整的轨迹 (`collect_trajectory`) 来计算并赋予每步转移（transition）正确的 Monte-Carlo return-to-go。  
	- 网络更新时，按照指定的混合比例（如 `mixing_ratio=0.5`）从固定的离线数据集和在线重放缓冲区中构建混合 Batch。  

## 支持的任务与环境

根据原作者开源的实现范围，本代码主要支持并内置了以下两类任务的官方预设参数：  

1. **D4RL AntMaze (稀疏迷宫任务)**
	- 默认预处理：`reward_scale=10.0`, `reward_bias=-5.0`, `sparse_failure_reward=-5.0`。  
	- 默认架构：Actor 隐藏层 256x2，Critic 隐藏层 256x4。  
	- 默认机制：开启拉格朗日乘子，目标间隙设为 0.8。  
2. **Adroit-Binary (灵巧手任务)**
	- 需提供包含官方演示数据的本地 `.npy` 目录路径。  
	- 默认预处理：`reward_scale=10.0`, `reward_bias=5.0`, `sparse_failure_reward=-5.0`。  

注意：Cal-QL 官方仓库未发布稠密奖励的 Locomotion（如 HalfCheetah）任务的轨迹返回估计器实现，论文中该类任务使用的是拟合的 SARSA Q 下界。因此，本代码默认禁用对 Locomotion 任务使用通用 MC 返回值的非官方扩展行为。  

## 依赖库

- `torch`

	  

- `gym`

	  

- `d4rl`

	  

- `numpy`

	  

- `pyrallis`

	  

- `tqdm`

	  

- `tensorboard`

	  

## 运行指令

代码使用 `pyrallis` 管理参数。`official_preset` 标志默认开启，会自动根据环境名称覆盖网络宽度、学习率及预处理参数以对齐官方实验。  

### 运行 AntMaze 任务

Bash

```
python calql_pytorch.py --env_name antmaze-medium-diverse-v2
```

### 运行 Adroit-Binary 任务

需显式指定数据源并提供官方数据文件路径：  

Bash

```
python calql_pytorch.py --env_name pen-binary-v0 --dataset_source adroit_binary --adroit_binary_data_dir /path/to/offpolicy_hand_data
```

## 关键参数列表

- `--env_name`: 目标环境名称（默认: `antmaze-medium-diverse-v2`）。  
- `--pretrain_steps`: 离线预训练阶段的梯度更新总步数。  
- `--num_total_steps`: 在线微调阶段的环境交互总步数。  
- `--mixing_ratio`: 在线更新时，从离线数据集中采样的数据所占的比例（默认: `0.5`）。  
- `--cql_min_q_weight`: CQL 保守惩罚项的基础权重。  
- `--enable_calql`: 是否开启 Q 值下界校准截断机制（默认: `True`）。  