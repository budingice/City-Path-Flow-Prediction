# 全时间步测试集预测结果对比脚本使用指南

## 📋 概述

`full_test_comparison.py` 脚本用于在测试集的全时间步上展示基准模型和动态模型的预测结果对比，生成完整的时序预测对比图表。

## 🚀 快速开始

### 基本用法

```bash
# 运行完整对比分析（默认只比较 HA_Baseline、LSTM_Baseline、adaptive_semantic_weighted）
python scripts/full_test_comparison.py

# 指定特定路径和时间步数
python scripts/full_test_comparison.py --path_idx 15 --max_time_steps 100

# 查看完整测试集（所有时间步）
python scripts/full_test_comparison.py --max_time_steps 0
```

## 📊 输出文件

脚本会在 `experiments/full_test_comparison/` 目录下生成以下文件：

### 1. 时序对比图 (`full_comparison_path{idx}_{steps}steps_{timestamp}.png`)
- **X轴**：时间步 (0 到指定步数)
- **Y轴**：交通流量 (vehicles/minute)
- **蓝色实线**：真实值 (Ground Truth)
- **彩色线条**：各模型预测值
  - 基准模型：HA_Baseline, Linear_Baseline, LSTM_Baseline
  - 动态模型：adaptive_semantic_base, adaptive_semantic_trend, 等
- **灰色区域**：无效数据区域

### 2. 性能对比图 (`metrics_comparison_{timestamp}.png`)
- **MAE, RMSE, WAPE** 三指标的柱状图对比
- 不同颜色区分基准模型和动态模型

### 3. 指标汇总 (`metrics_summary.json`)
```json
{
  "adaptive_semantic_base": {
    "mae": 0.509,
    "rmse": 1.461,
    "wape": 1.818
  },
  "HA_Baseline": {
    "mae": 0.254,
    "rmse": 1.089,
    "wape": 0.963
  }
}
```

### 4. 完整结果文件 (`{model_name}_full_results.npz`)
- 包含每个模型的完整预测结果、真实值和掩码

## 📈 模型说明

### 基准模型 (Baselines)
- **HA_Baseline**: 历史平均值预测
- **Linear_Baseline**: 线性时序回归
- **LSTM_Baseline**: 纯LSTM时序预测

### 动态模型 (Dynamic Models)
- **adaptive_semantic_base**: 基础自适应语义模型
- **adaptive_semantic_trend**: 趋势增强自适应语义模型
- **adaptive_semantic_weighted**: 加权自适应语义模型
- **adaptive_topo_base**: 基础自适应拓扑模型

## 🎨 可视化特性

- **字体**: 中文标题使用宋体，英文使用Times New Roman
- **分辨率**: 300 DPI 高清输出
- **尺寸**: 时序图 20x10 英寸，性能图 18x6 英寸
- **颜色方案**: 自动区分基准模型和动态模型
- **图例**: 清晰的模型类型标识

## 📐 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/config.yaml` | 配置文件路径 |
| `--path_idx` | `0` | 要分析的路径索引 (0-199) |
| `--max_time_steps` | `None` | 最大时间步数 (None=全部) |
| `--baseline_count` | `2` | 比较的基线模型数量 (建议 2-3) |
| `--dynamic_count` | `1` | 比较的动态模型数量 (建议 1-2) |

## 🔧 工作流程

1. **加载数据**: 从 `data/model_input/st_batch_data.pt` 加载测试数据
2. **加载模型**:
   - 优先从 `experiments/predictions/` 加载预计算结果
   - 其次从 `experiments/best_model/` 加载PyTorch模型
   - 最后初始化基准模型
3. **运行推理**: 对所有模型进行全测试集推理
4. **计算指标**: MAE, RMSE, WAPE 三指标
5. **生成图表**: 时序对比图和性能对比图
6. **保存结果**: 所有结果保存到 `experiments/full_test_comparison/`

## 💡 使用建议

### 分析特定路径
```bash
# 分析路径50的前2000个时间步
python scripts/full_test_comparison.py --path_idx 50 --max_time_steps 2000
```

### 比较不同模型性能
```bash
# 查看性能对比图
start experiments\full_test_comparison\metrics_comparison_*.png
```

### 深入分析结果
```python
import numpy as np

# 加载结果
data = np.load('experiments/full_test_comparison/adaptive_semantic_base_full_results.npz')
preds = data['preds']  # [样本数, 预测步数, 节点数]
trues = data['trues']  # [样本数, 预测步数, 节点数]
masks = data['masks']  # [样本数, 预测步数, 节点数]
```

## 🐛 故障排除

### 模型加载失败
- 检查 `experiments/predictions/` 是否有预计算结果
- 检查 `experiments/best_model/` 是否有PyTorch模型文件
- 基准模型会自动初始化，无需外部文件

### 内存不足
- 减小 `--max_time_steps` 参数
- 使用 `--path_idx` 只分析特定路径

### 中文字体显示问题
- Windows: 确保系统安装了宋体
- Linux: `sudo apt-get install fonts-noto-cjk`

## 📊 性能基准

基于当前结果的模型性能排序：

| 排名 | 模型 | MAE | RMSE | WAPE |
|------|------|-----|------|------|
| 1 | adaptive_semantic_weighted | 0.401 | 1.369 | 1.363 |
| 2 | adaptive_semantic_trend | 0.438 | 1.514 | 1.595 |
| 3 | adaptive_semantic_base | 0.509 | 1.461 | 1.818 |
| 4 | adaptive_topo_base | 0.520 | 1.522 | 1.868 |
| 基准 | HA_Baseline | 0.254 | 1.089 | 0.963 |

---

**最后更新**: 2025-01-14</content>
<parameter name="filePath">d:\毕业设计\路径流量预测\scripts\README_full_test_comparison.md