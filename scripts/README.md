# 🚀 AST-GCN 模型推理与验证脚本使用指南

## 📋 概述

本项目提供了完整的模型推理和验证工具集，包括：

1. **verify_model.py** - 完整的多模型对比推理脚本
2. **infer_single_model.py** - 单模型推理脚本
3. **test_verify_model.py** - 环境检查脚本

---

## 🔧 环境检查

### 第一步：验证环境

运行环境检查脚本，确保所有必要的文件和依赖都已正确配置：

```bash
python scripts/test_verify_model.py
```

**预期输出：** 所有 6 项测试都应该通过 (✅ PASS)

```
✅ PASS - 配置文件
✅ PASS - 模型权重
✅ PASS - 数据加载
✅ PASS - 模型加载
✅ PASS - 指标计算
✅ PASS - 输出目录

总体: 6/6 测试通过

✅ 所有测试通过！环境正常，可以运行推理脚本。
```

**常见问题排查：**

| 错误 | 解决方案 |
|------|--------|
| "配置文件不存在" | 检查 `configs/config.yaml` 是否存在 |
| "模型权重文件不存在" | 运行训练脚本生成 `experiments/best_model/best_model.pth` |
| "数据文件不存在" | 检查 `data/model_input/st_batch_data.pt` 是否存在 |
| "GPU 不可用" | 脚本会自动降低 CPU，继续运行 (可能较慢) |

---

## 🎯 快速开始

### 方案 A: 推理单个模型（推荐）

最快、最简单的选择 - 推理一个模型并计算指标：

```bash
python scripts/infer_single_model.py \
  --model_name adaptive_semantic_base \
  --adj_type semantic
```

**输出文件：**
- `experiments/predictions/adaptive_semantic_base_results.npz` - 预测结果
- `experiments/predictions/adaptive_semantic_base_metrics.json` - 评估指标

**示例输出：**
```
🚀 对 adaptive_semantic_base 进行推理...
  已处理 10 个 batch
  已处理 20 个 batch
  已处理 30 个 batch
✅ 推理完成！结果形状: (1000, 200, 3)

📊 评估指标:
  MAE:  0.508692 辆/分钟
  RMSE: 1.461348
  WAPE: 1.818432
```

---

### 方案 B: 多模型对比（完整分析）

对比多个模型，生成对比图表：

```bash
# 不包含基线模型
python scripts/verify_model.py \
  --model_name adaptive_semantic_base \
  --num_time_steps 500

# 包含基线模型（需要基线模型权重）
python scripts/verify_model.py \
  --model_name adaptive_semantic_base \
  --include_baselines \
  --num_time_steps 1000
```

**输出文件：**
- `experiments/verify_plots/comparison_sample0_path0.png` - 单步对比图
- `experiments/verify_plots/timeseries_path0_500steps.png` - 时序对比图
- `experiments/predictions/` - 所有模型的预测结果和指标

---

## 📊 详细参数说明

### infer_single_model.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | `adaptive_semantic_base` | 模型名称（用于文件保存） |
| `--model_path` | `experiments/best_model/best_model.pth` | 模型权重文件路径 |
| `--adj_type` | `semantic` | 邻接矩阵类型：`semantic` 或 `topo` |
| `--batch_size` | `32` | 推理批大小（GPU 内存不足时减小） |

**示例：**
```bash
# 使用物理拓扑邻接矩阵
python scripts/infer_single_model.py --adj_type topo

# 使用较小批大小（GPU 内存不足）
python scripts/infer_single_model.py --batch_size 8

# 指定自定义模型路径
python scripts/infer_single_model.py --model_path checkpoints/my_model.pth
```

---

### verify_model.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/config.yaml` | 配置文件路径 |
| `--model_dir` | `experiments/best_model` | 模型保存目录 |
| `--model_name` | `adaptive_semantic_base` | 模型名称 |
| `--sample_idx` | `0` | 用于绘图的样本索引 |
| `--path_idx` | `0` | 用于绘图的路径索引（0-199） |
| `--num_time_steps` | `500` | 时序图显示的时间步数 |
| `--adj_type` | `semantic` | 邻接矩阵类型 |
| `--batch_size` | `32` | 推理批大小 |
| `--include_baselines` | 否 | 是否包含基线模型对比 |

**示例：**
```bash
# 基础推理
python scripts/verify_model.py

# 查看特定路径的预测
python scripts/verify_model.py --path_idx 50 --sample_idx 5

# 长时间序列对比
python scripts/verify_model.py --num_time_steps 2000

# 包含所有基线模型
python scripts/verify_model.py --include_baselines

# 自定义配置和模型路径
python scripts/verify_model.py \
  --config my_config.yaml \
  --model_dir checkpoints/my_exp
```

---

## 📈 输出文件说明

### 推理结果文件

#### NPZ 格式 (experiments/predictions/{model_name}_results.npz)

使用 numpy 读取：
```python
import numpy as np

# 加载结果
data = np.load('experiments/predictions/adaptive_semantic_base_results.npz')

# 获取三个数组
predictions = data['preds']      # [样本数, 节点数, 预测步数]
ground_truth = data['trues']     # [样本数, 节点数, 预测步数]
masks = data['masks']             # [样本数, 节点数, 预测步数]

print(f"预测形状: {predictions.shape}")
print(f"样本 0，路径 0 的预测: {predictions[0, 0, :]}")
```

#### JSON 指标文件 (metrics_summary.json)

```json
{
  "adaptive_semantic_base": {
    "mae": 0.508692,
    "rmse": 1.461348,
    "wape": 1.818432
  },
  "HA_Baseline": {
    "mae": 0.254070,
    "rmse": 1.088889,
    "wape": 0.963421
  }
}
```

### 图表文件说明

#### 1. 单步预测对比图 (comparison_sample{idx}_path{idx}.png)

- **蓝色实线**：真实值 (Ground Truth)
- **彩色虚线/点划线**：各模型的预测值
- **X 轴**：预测步数 (1-3)
- **Y 轴**：交通流量 (辆/分钟)

#### 2. 时序预测对比图 (timeseries_path{idx}_{steps}steps.png)

- **蓝色实线**：完整时序的真实值
- **彩色线条**：各模型的滚动预测值
- **灰色虚线**：无效数据区域 (掩码为 0)
- **X 轴**：时间步 (0 到指定的步数)
- **Y 轴**：交通流量 (辆/分钟)

---

## 🎨 可视化说明

### 字体配置

- **中文标题**：宋体 (SimSun)
- **英文和数字**：Times New Roman
- **标题字号**：14pt (粗体)
- **坐标轴标签**：13pt (粗体)
- **图例**：11pt

### 线条样式

| 模型 | 颜色 | 线型 | 标记 |
|------|------|------|------|
| 真实值 | 蓝色 | 实线 (-) | 圆形 (o) |
| 预测值 1 | 红色 | 虚线 (--) | 方形 (s) |
| 预测值 2 | 绿色 | 点划线 (-.) | 方形 (s) |
| 预测值 3 | 橙色 | 虚线 (:) | 方形 (s) |

---

## 📐 评估指标解释

### MAE (Mean Absolute Error) - 平均绝对误差

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **单位**：辆/分钟
- **范围**：0 到无穷大
- **解释**：平均预测误差，越小越好
- **优点**：易于理解，对异常值敏感度较低

### RMSE (Root Mean Squared Error) - 均方根误差

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **单位**：辆/分钟
- **范围**：0 到无穷大
- **解释**：对较大误差的惩罚更重，越小越好
- **优点**：对异常值更敏感，适合检测极端情况

### WAPE (Weighted Absolute Percentage Error) - 加权绝对百分比误差

$$\text{WAPE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|}$$

- **单位**：无（百分比形式）
- **范围**：0 到无穷大（理论上可以 > 1）
- **解释**：预测误差占真实值总和的比例，越小越好
- **优点**：适合比较不同规模数据集的模型性能

---

## 🔄 工作流程示例

### 完整分析流程

```bash
# 1. 检查环境
python scripts/test_verify_model.py

# 2. 推理主模型
python scripts/infer_single_model.py \
  --model_name adaptive_semantic_base

# 3. 生成对比图表（包含基线）
python scripts/verify_model.py \
  --model_name adaptive_semantic_base \
  --include_baselines \
  --sample_idx 0 \
  --path_idx 0

# 4. 查看多个路径的结果
for path_id in 0 10 25 50; do
  python scripts/verify_model.py \
    --path_idx $path_id \
    --num_time_steps 300
done

# 5. 检查结果
ls -lh experiments/verify_plots/
cat experiments/predictions/metrics_summary.json
```

### Python 编程接口

如果需要在自己的代码中调用推理函数：

```python
import sys
import torch
import yaml
from src.models import STGCN_LSTM_Adaptive, TrafficDataset
from torch.utils.data import DataLoader

# 加载配置和数据
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset = TrafficDataset(
    config['path']['model_input_pt'],
    window_size=config['train']['window_size'],
    horizon=config['train']['horizon']
)

# 创建和加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STGCN_LSTM_Adaptive(
    adj=dataset.adj,
    num_nodes=dataset.adj.shape[0],
    hidden_dim=config['model']['hidden_dim'],
    horizon=config['train']['horizon']
)
model.load_state_dict(torch.load('experiments/best_model/best_model.pth'))
model = model.to(device)
model.eval()

# 推理
test_loader = DataLoader(dataset, batch_size=32)
with torch.no_grad():
    for x, y, mask in test_loader:
        x = x.to(device)
        pred = model(x, torch.tensor(dataset.adj, device=device))
        # 处理预测结果...
```

---

## 🐛 常见问题解决

### Q1: CUDA 内存不足

**错误信息：** `RuntimeError: CUDA out of memory`

**解决方案：**
```bash
# 减小批大小
python scripts/infer_single_model.py --batch_size 8

# 或使用 CPU
CUDA_VISIBLE_DEVICES="" python scripts/infer_single_model.py
```

### Q2: 中文字体显示乱码

**症状：** 图表标题显示为方框

**解决方案：**
- Windows：确保系统安装了宋体字体（通常默认有）
- Linux：安装宋体字体包
  ```bash
  sudo apt-get install fonts-noto-cjk
  ```
- macOS：使用系统自带的宋体或下载安装

### Q3: 基线模型加载失败

**错误信息：** `⚠️ 加载 {baseline_name} 失败`

**原因：** 基线模型权重文件不存在

**解决方案：**
- 查阅 `experiments/predictions/` 目录中是否有其他模型的权重
- 或直接调用对应的基线模型类（不需要权重文件）

### Q4: 推理速度很慢

**原因：** 可能使用 CPU 而非 GPU

**检查方法：**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**解决方案：**
- 确保已安装 CUDA 和 cuDNN
- 检查 PyTorch 是否正确编译了 CUDA 支持

---

## 📚 性能基准参考

根据 `experiments/benchmark_summary.csv` 的汇总结果：

| 模型 | MAE | RMSE | WAPE |
|------|-----|------|------|
| **最佳模型** |
| adaptive_semantic_weighted | 0.401 | 1.369 | 1.363 |
| adaptive_semantic_trend | 0.438 | 1.514 | 1.595 |
| **改进模型** |
| adaptive_semantic_base | 0.509 | 1.461 | 1.818 |
| adaptive_topo_base | 0.520 | 1.522 | 1.868 |
| **强基线** |
| HA_Baseline | 0.254 | 1.089 | 0.963 |
| Standard_STGCN | 0.545 | 1.484 | 1.876 |
| **其他模型** |
| LSTM_Baseline | 0.485 | 1.356 | 2.152 |
| Linear_Baseline | 1.427 | 4.182 | 5.491 |

---

## 🎯 下一步

1. **生成论文图表**：使用 `verify_model.py` 生成高质量的对比图
2. **分析模型特性**：查看 `experiments/predictions/` 中的详细结果
3. **进一步优化**：基于评估结果调整超参数
4. **发布结果**：将汇总的指标表格添加到论文

---

## 📞 技术支持

如遇到问题，请：
1. 检查此文档中的 FAQ 部分
2. 运行 `test_verify_model.py` 进行环境诊断
3. 查看脚本输出中的详细错误信息
4. 检查日志文件 `experiments/` 目录

---

## 📄 许可证

此脚本套件作为毕业设计项目的一部分发布。

---

**最后更新**：2025-05-14
**版本**：1.0
