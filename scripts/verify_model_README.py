"""
========================================================================================
AST-GCN 模型推理与验证 - 快速开始指南
========================================================================================

本脚本用于：
  ✅ 加载训练好的 AST-GCN 模型
  ✅ 在测试集上运行推理
  ✅ 计算 MAE、RMSE、WAPE 评估指标
  ✅ 生成预测值 vs 真实值的对比图表
  ✅ 与其他基线模型进行对比分析

========================================================================================
使用方法
========================================================================================

1. 基础用法 - 推理最优模型并生成对比图：
   
   python scripts/verify_model.py \\
     --model_name adaptive_semantic_base \\
     --adj_type semantic \\
     --sample_idx 0 \\
     --path_idx 0
   

2. 包含基线模型对比：
   
   python scripts/verify_model.py \\
     --model_name adaptive_semantic_base \\
     --include_baselines \\
     --num_time_steps 500
   

3. 自定义配置：
   
   python scripts/verify_model.py \\
     --config configs/config.yaml \\
     --model_dir experiments/best_model \\
     --batch_size 64 \\
     --num_time_steps 1000
   

========================================================================================
参数说明
========================================================================================

必选参数：
  无（所有参数都有默认值）

可选参数：
  --config PATH              配置文件路径 (默认: configs/config.yaml)
  --model_dir PATH           模型保存目录 (默认: experiments/best_model)
  --model_name NAME          模型名称 (默认: adaptive_semantic_base)
                            用于标识在结果表中的显示名称
  --sample_idx IDX           用于绘图的样本索引 (默认: 0)
  --path_idx IDX             用于绘图的路径（节点）索引 (默认: 0)
  --num_time_steps N         时序图显示的时间步数 (默认: 500)
  --adj_type TYPE            邻接矩阵类型 (默认: semantic)
                            选项: semantic, topo
  --batch_size N             推理批大小 (默认: 32)
  --include_baselines        是否包含基线模型进行对比 (标志位)

========================================================================================
输出文件说明
========================================================================================

所有输出文件保存在 experiments/verify_plots/ 目录：

1. 对比图表：
   ├─ comparison_sample{idx}_path{idx}.png
   │  └─ 单步预测对比图，包含所有模型的预测结果
   │
   └─ timeseries_path{idx}_{steps}steps.png
      └─ 长时间跨度的时序预测对比图

2. 详细预测结果 (experiments/predictions/):
   ├─ {model_name}_predictions.npz
   │  └─ 包含 predictions, trues, masks 三个数组
   │
   └─ metrics_summary.json
      └─ 所有模型的指标汇总 (MAE, RMSE, WAPE)

========================================================================================
输出指标说明
========================================================================================

1. MAE (Mean Absolute Error) - 平均绝对误差
   公式: MAE = mean(|pred - true|)
   单位: 辆/分钟
   越小越好

2. RMSE (Root Mean Squared Error) - 均方根误差
   公式: RMSE = sqrt(mean((pred - true)^2))
   单位: 辆/分钟
   对大误差的惩罚更重，越小越好

3. WAPE (Weighted Absolute Percentage Error) - 加权绝对百分比误差
   公式: WAPE = sum(|pred - true|) / sum(|true|)
   无单位（百分比）
   对整体预测准确度的衡量，越小越好

========================================================================================
绘图样式说明
========================================================================================

所有绘图采用学术风格：

◆ 真实值 (Ground Truth):
  - 颜色: 蓝色 (Blue)
  - 线型: 实线 (-)
  - 标记: 圆形 (o)
  - 线宽: 2.5pt

◆ 预测值:
  - 颜色: 红色及其他颜色
  - 线型: 虚线 (--), 点划线 (-.), 等
  - 标记: 方形 (s)
  - 线宽: 2pt
  - 透明度: 80%

◆ 字体:
  - 中文标题: 宋体 (SimSun)
  - 英文和数字: Times New Roman
  - 标题字号: 14pt (粗体)
  - 坐标轴标签: 13pt (粗体)
  - 图例: 11pt

◆ 网格:
  - 类型: 虚线 (:)
  - 透明度: 40%

========================================================================================
常见问题
========================================================================================

Q1: "找不到模型文件" 错误

A: 确保 experiments/best_model/best_model.pth 文件存在
   可以运行: ls experiments/best_model/
   如果没有，需要先执行训练脚本

Q2: 如何查看其他路径的预测结果？

A: 修改 --path_idx 参数
   例: --path_idx 50  (查看第 50 号路径)
   项目中共有 200 条路径（0-199）

Q3: 如何只看特定样本的预测？

A: 修改 --sample_idx 参数
   例: --sample_idx 10  (查看第 10 个样本)

Q4: 如何对比不同模型？

A: 使用 --include_baselines 标志位
   这会自动加载并对比 4 个基线模型

Q5: GPU 内存不足？

A: 减少 --batch_size 值
   例: --batch_size 8 或 --batch_size 16

Q6: 绘图显示中文乱码？

A: 已配置 matplotlib 使用宋体字体
   如果仍有问题，检查系统是否安装了宋体字体
   Windows 系统默认应该有

========================================================================================
示例代码片段
========================================================================================

# 1. 推理单个模型
python scripts/verify_model.py --model_name adaptive_semantic_base

# 2. 与基线模型对比
python scripts/verify_model.py \\
  --model_name adaptive_semantic_base \\
  --include_baselines \\
  --num_time_steps 1000

# 3. 查看多个路径的预测结果
for path_idx in 0 10 25 50 100; do
    python scripts/verify_model.py \\
      --model_name adaptive_semantic_base \\
      --path_idx $path_idx
done

# 4. 使用 topo 邻接矩阵
python scripts/verify_model.py \\
  --model_name adaptive_topo_base \\
  --adj_type topo

========================================================================================
性能基准 (预期结果)
========================================================================================

根据 benchmark_summary.csv 的统计：

优秀模型:
  - adaptive_semantic_base: MAE ≈ 0.509, RMSE ≈ 1.461, WAPE ≈ 1.818
  - adaptive_semantic_weighted: MAE ≈ 0.401, RMSE ≈ 1.369, WAPE ≈ 1.363

基线模型:
  - HA (Historical Average): MAE ≈ 0.254, RMSE ≈ 1.089, WAPE ≈ 0.963
  - LSTM Baseline: MAE ≈ 0.485, RMSE ≈ 1.356, WAPE ≈ 2.152
  - Standard STGCN: MAE ≈ 0.545, RMSE ≈ 1.484, WAPE ≈ 1.876

========================================================================================
"""

if __name__ == '__main__':
    print(__doc__)
