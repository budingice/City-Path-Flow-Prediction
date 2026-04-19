# Urban Path Flow Prediction based on STGCN-LSTM
# 基于深度学习的城市路径流量预测

[cite_start]本项目为本人的毕业设计项目。旨在利用高分辨率无人机轨迹数据（pNEUMA），通过融合图卷积网络（GCN）与长短期记忆网络（LSTM），实现对城市交通路径流量的精准预测。 [cite: 11, 15]

## 🌟 项目亮点
- [cite_start]**数据源**： 基于pNEUMA公开数据集，该数据集为雅典城区无人机采集的车辆轨迹，频率1Hz，涵盖 13,973 条完整轨迹。 [cite: 119, 121]
- [cite_start]**空间相关性建模**：基于路径间共享路段的 Jaccard 相似度构建关联矩阵（Path Correlation Matrix），捕捉路径间的空间竞争。 [cite: 6]
- [cite_start]**时空融合模型**：基于**STGCN-LSTM** 架构，其中 GCN 提取空间拓扑特征，LSTM 学习时间维度演变规律。 [cite: 6]

## 🏗️ 模型架构
[cite_start]模型主要由以下四部分组成： [cite: 6]
1. **GCN层**：捕捉路径间的空间相关性，将流量特征映射至高维特征向量。
2. **LSTM层**：经典的 LSTM 结构，负责学习流量在时间轴上的变化规律。
3. **输出层**：通过全连接层进行维度变换，输出预测流量。
4. **损失函数**：采用均方误差（MSE）作为训练目标。

## 📊 阶段性成果
[cite_start]模型在雅典区域 1（含 34 条核心路段、50 条 Top 路径）上进行了实验验证： [cite: 119, 6]
- [cite_start]**总体性能**：平均 MAE 0.5246，平均 RMSE 0.7780。 [cite: 7]
- **多步预测**：
  - [cite_start]T+1 (5min): R² = 0.5265 [cite: 7]
  - [cite_start]T+2 (10min): R² = 0.4315 [cite: 7]
- [cite_start]**数据覆盖**：选取的 Top 50 核心路径已足以代表研究区域 68.95% 的交通分布。 [cite: 6]

## 🛠️ 技术栈
- [cite_start]**语言**：Python 3.12, R 4.5.3 [cite: 93]
- [cite_start]**深度学习**：PyTorch [cite: 94]
- **地理信息/分析**：Pandas, NumPy, SciPy, R-Tree
- **可视化**：Matplotlib, Seaborn, Folium

## 🚀 运行示例
### 1. 数据对齐与路径统计
运行 `analyze_path_competition_map.py` 进行 OD 对提取与路径竞争分析：
```powershell
# 统计 Top 10 OD 对及 Top 5 路径地图分布
python analyze_path_competition_map.py --top-n-od 10 --top-map-ods 5 --min-flow 5