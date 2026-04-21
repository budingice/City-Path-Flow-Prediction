# 📋 城市路径流量预测：消融实验运行手册 (Experiment Manual)

> **当前课题**：Urban Path Flow Prediction  
> **目标**：对比静态邻接矩阵 (Static Jaccard) 与自适应邻接矩阵 (Adaptive) 对预测精度的影响。

---

## 📂 1. 项目架构说明 (Project Layout)

采用模块化设计，将模型、训练、评估逻辑解耦，确保实验的严谨性。

```text
Urban-Path-Flow/
├── model_inputs/           # [输入] 存放 st_batch_data.pt 等原始数据
├── model_results/          # [输出] 存放实验原始数据 (.npz) 与指标 (.txt)
├── plots/                  # [展示] 存放对比曲线、热力图等可视化结果
├── models.py               # [核心] 定义不同的 STGCN 架构 (Static/Adaptive)
├── trainer.py              # [引擎] 通用的训练、保存、评估函数
├── run_ablation.py         # [启动] 实验的总控开关，一键运行对比实验
├── evaluator.py            # [分析] 独立的绘图与数据分析工具
└── step5_dataset1.py       # [数据] 现有的 TrafficDataset 定义类

## 🛠️ 2. 核心模块代码实现

## 3.运行指南
第一步：一键运行消融实验
运行 python run_ablation.py。该脚本会：

自动调用 trainer.py 进行静态模型训练。

结束后自动切换至自适应模型训练。

所有的指标和预测结果都会被自动分类存入 model_results/。

第二步：生成可视化报告
运行 python evaluator.py。该脚本会：

从 model_results/ 加载不同模型的预测数据。

在 plots/ 下生成对比图表，直接用于毕业论文。
