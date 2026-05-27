# 基于深度学习的城市路径流量预测

Urban Path Flow Prediction Based on Deep Learning

本仓库为本科毕业设计《基于深度学习的城市路径流量预测》的代码与实验材料。项目基于 pNEUMA 无人机车辆轨迹数据，从原始 GPS 轨迹出发完成轨迹清洗、道路匹配、路径流量聚合，并构建融合拓扑邻接与语义邻接的自适应时空图预测模型，用于预测未来多时间步的城市路径级交通流量。

## 研究目标

路径流量反映 OD 出行需求在路网中的空间分布，能够为信号优化配时、车辆诱导和交通状态研判提供支撑。相比路段级流量，路径级流量天然携带 OD 属性、转向关系和拥堵传播信息，但也面临路网拓扑复杂、轨迹缺失、地图匹配误差和路径高波动性等问题。

本项目的核心目标是建立一条从“全域车辆轨迹观测”到“路径级流量预测”的技术流程，并验证自适应时空图模型在路径流量预测任务中的有效性。

## 方法概览

整体流程包括四个部分：

1. **轨迹到路径流量的数据治理**
   使用 pNEUMA 轨迹数据，完成 GPS 点位清洗、采样频率调整、R-tree 空间索引检索、道路匹配和路径序列提取。

2. **路径流量矩阵构建**
   按出发时间定义路径流量，以固定时间步聚合同一路径上的车辆数，并引入 mask 通道区分“真实低流量”和“无观测缺失”。

3. **自适应时空图预测模型**
   使用 GCN 提取路径间空间关联，使用 LSTM 捕获时间依赖，并融合两类邻接信息：
   - 拓扑邻接：基于路径共享路段的 Jaccard 相似度。
   - 语义邻接：基于可学习节点嵌入的流量变化相似性。

4. **模型训练与实验验证**
   通过基线对比、邻接矩阵消融实验、不同波动性路径分析和 50min 预测结果对比，评估模型在 MAE、RMSE、WAPE 等指标上的表现。

## 主要结论

根据论文终稿与终期答辩材料，实验表明：

- Top 50 核心路径覆盖约 68.95% 的研究区域交通分布，可代表主要路径流量结构。
- 在对比实验中，自适应图模型取得较优表现：MAE 约 0.401，RMSE 约 1.369，WAPE 约 1.36%。
- 在邻接矩阵消融实验中，拓扑与语义融合的 Adaptive 模型优于单一拓扑邻接和单一语义邻接，说明路径共享路段关系与流量协同关系具有互补性。
- 对高波动路径，自适应模型能够缓解单一邻接信息失效带来的误差，但长时预测中仍存在一定滞后。

## 创新点

- **从单车轨迹到路径级流量数据**：直接利用全域轨迹观测构建路径流量矩阵，而不是仅依赖路段检测器或 OD 反推。
- **拓扑与语义邻接加权融合**：同时利用路径共享路段关系和流量趋势协同关系，动态调整图结构表达。
- **面向缺失观测的 mask 机制**：在输入和损失函数中显式区分低流量与缺失数据，降低虚假零值对模型训练的干扰。
- **路径级预测视角**：将交通预测对象从局部路段扩展到具有 OD 与转向含义的路径节点。

## 仓库结构

```text
.
├── configs/                 # 训练、数据路径和模型超参数配置
├── data/                    # 原始/处理后数据与模型输入文件（大文件不建议提交）
├── scripts/                 # 训练、推理、验证、绘图和参数寻优脚本
├── src/
│   ├── data_utils/          # 数据清洗、道路匹配、特征构建、质量分析
│   ├── models/              # Adaptive STGCN、Vanilla STGCN 与基线模型
│   ├── outputs/             # 结果可视化工具
│   ├── training/            # 训练器、损失函数与评估逻辑
│   └── utils/               # 配置读取、评价和辅助工具
├── train_entry.py           # 统一训练入口
├── run_test.py              # 快速测试入口
├── COMPLETION_REPORT.md     # 终稿完成报告
├── COMPLETION_SUMMARY.md    # 终稿摘要
└── QUICK_REFERENCE.md       # 快速参考
```

## 环境配置

推荐使用 Python 3.12。主要依赖包括 PyTorch、pandas、NumPy、PyYAML、pyarrow、osmnx、networkx、geopandas、shapely、matplotlib、seaborn、scikit-learn 等。

```bash
pip install -r configs/requirements.txt
```

如需使用 GPU，请根据本机 CUDA 版本安装匹配的 PyTorch 版本。

## 运行方式

### 1. 检查配置

核心配置位于：

```text
configs/config.yaml
```

需要重点检查：

- `path.model_input_pt`：模型输入张量路径，默认 `data/model_input/st_batch_data.pt`
- `train.window_size`：历史窗口长度
- `train.horizon`：预测步长
- `train.lr`、`model.hidden_dim`：训练超参数
- `ablation`：mask、自适应邻接、拓扑融合等消融开关

### 2. 训练模型

训练自适应模型：

```bash
python train_entry.py --model_type adaptive --adj_type semantic --loss_mode base
```

训练其他基线模型：

```bash
python train_entry.py --model_type ha
python train_entry.py --model_type linear
python train_entry.py --model_type lstm
python train_entry.py --model_type standard_stgcn
python train_entry.py --model_type vanilla --adj_type topo
```

指定超参数：

```bash
python train_entry.py --model_type adaptive --hidden_dim 220 --lr 0.000156 --exp_group tuning
```

### 3. 模型验证与推理

运行环境检查：

```bash
python scripts/test_verify_model.py
```

单模型推理：

```bash
python scripts/infer_single_model.py --model_name adaptive_semantic_base --adj_type semantic
```

多模型对比与可视化：

```bash
python scripts/verify_model.py --model_name adaptive_semantic_base --include_baselines --num_time_steps 1000
```

更多脚本说明见 [scripts/README.md](scripts/README.md)。

## 数据说明

项目使用 pNEUMA 数据集中的雅典中心城区无人机车辆轨迹数据。答辩材料中以区域 1 为例，路网包含 663 个节点、1184 条边，其中核心研究路段 34 条，轨迹片段总数约 13,973 条。

由于数据、模型权重和实验输出体积较大，仓库通过 `.gitignore` 忽略了 `.parquet`、`.pt`、`.pth`、`.npz`、实验图表和可视化输出。复现实验时需在本地准备对应数据文件。

## 参考材料

README 内容根据以下终稿材料整理：

- `2252749-冯乙潇-终期答辩.pptx`
- `2526_31_10247_081802_2252749_LW.docx`

论文题目：基于深度学习的城市路径流量预测  
作者：冯乙潇  
指导老师：朱宏  
完成时间：2026 年 5 月
