# 🎯 AST-GCN 模型推理脚本 - 快速参考卡片

## 📦 已创建的文件

| 文件 | 类型 | 大小 | 功能 |
|------|------|------|------|
| verify_model.py | 🐍 脚本 | ~800 行 | ⭐ **主脚本** - 完整推理和多模型对比 |
| infer_single_model.py | 🐍 脚本 | ~300 行 | 快速单模型推理 |
| test_verify_model.py | 🐍 脚本 | ~400 行 | 环境检查和诊断 |
| quick_run.bat | 批处理 | ~80 行 | Windows 快速启动 |
| quick_run.sh | Shell 脚本 | ~120 行 | Linux/Mac 快速启动 |
| README.md | 📖 文档 | ~600 行 | **完整使用指南** |
| COMPLETION_SUMMARY.md | 📖 文档 | ~400 行 | 项目完成总结 |

---

## ⚡ 30 秒快速启动

### Windows
```bash
cd d:\毕业设计\路径流量预测
scripts\quick_run.bat infer
```

### Linux/Mac
```bash
cd ~/路径流量预测
chmod +x scripts/quick_run.sh
./scripts/quick_run.sh infer
```

### 手动运行
```bash
python scripts/infer_single_model.py --model_name adaptive_semantic_base
```

---

## 🎯 常用命令

### 单模型推理（推荐入门）
```bash
python scripts/infer_single_model.py
```
**输出：** 评估指标 + NPZ 预测结果

### 多模型对比（完整分析）
```bash
python scripts/verify_model.py --include_baselines
```
**输出：** 指标表格 + 对比图表

### 查看特定路径
```bash
python scripts/verify_model.py --path_idx 50 --num_time_steps 500
```
**输出：** 该路径的时序预测对比

### 环境检查
```bash
python scripts/test_verify_model.py
```
**输出：** 环境诊断报告

---

## 📊 输出文件位置

```
experiments/
├── verify_plots/           ← 图表文件
│   ├── comparison_*.png    ← 单步对比图
│   └── timeseries_*.png    ← 时序对比图
└── predictions/            ← 预测结果
    ├── *_predictions.npz   ← 详细预测
    ├── *_metrics.json      ← 指标值
    └── metrics_summary.json ← 汇总表
```

---

## 📈 预期指标范围

| 指标 | 优秀 | 良好 | 一般 |
|------|------|------|------|
| MAE | < 0.5 | 0.5-1.0 | > 1.0 |
| RMSE | < 1.5 | 1.5-2.0 | > 2.0 |
| WAPE | < 2.0 | 2.0-3.0 | > 3.0 |

**参考值：** adaptive_semantic_weighted (最优) = MAE: 0.401, RMSE: 1.369, WAPE: 1.363

---

## 🔧 常用参数

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `--model_name` | adaptive_semantic_base | 模型名称 |
| `--adj_type` | semantic | 邻接矩阵：semantic/topo |
| `--path_idx` | 0 | 查看第 N 条路径 (0-199) |
| `--sample_idx` | 0 | 查看第 N 个样本 |
| `--num_time_steps` | 500 | 时序图长度 |
| `--batch_size` | 32 | GPU 内存不足时减小 |
| `--include_baselines` | False | 包含基线模型对比 |

---

## ⚠️ 常见问题

### GPU 内存不足
```bash
python scripts/infer_single_model.py --batch_size 8
```

### 中文乱码
✅ 自动配置了宋体字体，通常无需修改

### 推理很慢
确认是否使用了 GPU：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 模型文件缺失
运行诊断脚本：
```bash
python scripts/test_verify_model.py
```

---

## 📚 查阅文档

| 场景 | 推荐文档 |
|------|--------|
| 第一次使用 | 👉 **README.md** |
| 脚本参数详解 | 👉 **README.md** (参数说明部分) |
| 使用示例 | 👉 **verify_model_README.py** (打印输出) |
| 故障排查 | 👉 **README.md** (FAQ 部分) |
| 项目总结 | 👉 **COMPLETION_SUMMARY.md** |

---

## 🎯 典型工作流

```
1. 环境检查
   └─→ python scripts/test_verify_model.py

2. 快速推理
   └─→ python scripts/infer_single_model.py

3. 生成对比图表
   └─→ python scripts/verify_model.py --include_baselines

4. 提取结果用于论文
   └─→ 从 experiments/verify_plots/ 复制图表
       从 experiments/predictions/ 提取指标
```

---

## 💡 高级用法

### 批量推理多个路径
```bash
for path_id in {0..199..10}; do
  python scripts/verify_model.py --path_idx $path_id &
done
wait
```

### 与论文中的基准对比
```bash
# 检查是否达到预期性能
python scripts/infer_single_model.py --model_name adaptive_semantic_weighted
# 应该看到: MAE ≈ 0.401, RMSE ≈ 1.369
```

### 自定义模型路径
```bash
python scripts/infer_single_model.py \
  --model_path checkpoints/my_best_model.pth \
  --model_name my_model
```

---

## 📞 获取帮助

1. **快速问题** → 查看此卡片
2. **详细问题** → 阅读 README.md
3. **环境问题** → 运行 test_verify_model.py
4. **使用示例** → 查看 verify_model_README.py

---

## ✅ 验收清单

- [x] 主推理脚本完成
- [x] 单模型推理脚本完成
- [x] 环境检查脚本完成
- [x] 快速启动脚本完成
- [x] 完整使用文档完成
- [x] 指标计算正确
- [x] 图表生成正确
- [x] 多模型对比支持
- [x] 中英文字体支持
- [x] 错误处理完善

---

## 🎓 论文使用建议

### 推荐的图表生成流程
```bash
# 1. 生成高质量对比图
python scripts/verify_model.py \
  --model_name adaptive_semantic_weighted \
  --sample_idx 0 \
  --path_idx 0

# 2. 生成时序图
python scripts/verify_model.py \
  --include_baselines \
  --num_time_steps 500

# 3. 从以下位置提取图表
# experiments/verify_plots/comparison_*.png
# experiments/verify_plots/timeseries_*.png
```

### 推荐指标表格
```
来自 experiments/predictions/metrics_summary.json
复制到论文的表格中
```

---

**最后更新**: 2025-05-14  
**版本**: 1.0  
**状态**: ✅ 完成并可用  

---

## 🚀 现在就开始！

```bash
python scripts/infer_single_model.py
```

该命令将：
1. ✅ 加载模型
2. ✅ 加载数据
3. ✅ 运行推理
4. ✅ 计算指标
5. ✅ 保存结果

**预计耗时**: 2-5 分钟 (GPU) / 10-20 分钟 (CPU)
