"""
========================================================================================
模型推理与验证脚本 - AST-GCN 流量预测模型
========================================================================================
功能：
  1. 加载训练好的最优模型
  2. 在测试集上运行推理
  3. 反归一化预测结果
  4. 计算 MAE, RMSE, WAPE 指标
  5. 绘制预测值 vs 真实值的对比图
  6. 与其他基线模型进行对比分析

使用方法：
  python scripts/verify_model.py --model_name adaptive_semantic_base --sample_idx 0
  
========================================================================================
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# 确保导入项目模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from src.models import STGCN_LSTM_Adaptive, VanillaSTGCN, TrafficDataset
from src.models.baselines import HA_Baseline, Linear_Baseline, LSTM_Baseline, Standard_STGCN


# ========================================================================================
# 核心函数 1: 模型加载
# ========================================================================================
def load_model(model_path, model_type='adaptive', adj_matrix=None, num_nodes=200, device='cuda'):
    """
    加载预训练的模型权重
    
    Args:
        model_path: 模型权重文件路径 (.pth)
        model_type: 模型类型 (adaptive, vanilla, ha, linear, lstm, standard_stgcn)
        adj_matrix: 邻接矩阵 (某些模型需要)
        num_nodes: 节点数
        device: 计算设备
        
    Returns:
        model: 加载权重后的模型
    """
    print(f"🔄 正在加载模型: {model_path}")
    
    # 根据模型类型创建模型实例
    if model_type == 'adaptive':
        model = STGCN_LSTM_Adaptive(
            adj=adj_matrix.cpu().numpy() if torch.is_tensor(adj_matrix) else adj_matrix,
            num_nodes=num_nodes,
            hidden_dim=220,
            horizon=3
        )
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type == 'vanilla':
        model = VanillaSTGCN(
            adj=adj_matrix.cpu().numpy() if torch.is_tensor(adj_matrix) else adj_matrix,
            num_nodes=num_nodes,
            hidden_dim=64,
            horizon=3
        )
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type == 'ha':
        model = HA_Baseline(horizon=3)
    elif model_type == 'linear':
        model = Linear_Baseline(window_size=12, horizon=3, num_nodes=num_nodes)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type == 'lstm':
        model = LSTM_Baseline(num_nodes=num_nodes, hidden_dim=64, horizon=3)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type == 'standard_stgcn':
        model = Standard_STGCN(num_nodes=num_nodes, hidden_dim=64, horizon=3)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"❌ 不支持的模型类型: {model_type}")
    
    # 移动到设备并设置评估模式
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()
    
    print(f"✅ 模型加载成功！模型类型: {model_type}")
    return model


# ========================================================================================
# 核心函数 2: 推理
# ========================================================================================
def inference(model, test_loader, adj_matrix=None, device='cuda', max_val=1.0, model_type='adaptive'):
    """
    在测试集上运行推理，返回预测结果和真实值
    
    Args:
        model: 已加载的模型
        test_loader: 测试集数据加载器
        adj_matrix: 邻接矩阵 (某些模型需要)
        device: 计算设备
        max_val: 规范化时使用的最大值
        model_type: 模型类型
        
    Returns:
        all_preds: 预测结果 [样本数, 节点数, 预测步数]
        all_trues: 真实值 [样本数, 节点数, 预测步数]
        all_masks: 掩码 [样本数, 节点数, 预测步数]
    """
    print(f"🚀 开始推理 (设备: {device})")
    
    all_preds = []
    all_trues = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 处理不同的数据格式
            if len(batch) == 3:
                x, y, mask = batch
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
            else:
                x = batch[0].to(device)
                y = batch[1].to(device)
                mask = batch[2].to(device) if len(batch) > 2 else torch.ones_like(batch[1])
            
            # 运行推理
            if isinstance(model, torch.nn.Module):
                if model_type in ['standard_stgcn', 'vanilla', 'adaptive']:
                    pred = model(x, adj_matrix)
                else:
                    try:
                        pred = model(x, adj_matrix)
                    except TypeError:
                        pred = model(x)
            else:
                # 对于非 Torch 模型（如 HA_Baseline）
                pred = torch.tensor(model.predict(x), dtype=torch.float32, device=device)
            
            # 反归一化
            pred_denorm = pred.cpu().numpy() * max_val
            y_denorm = y.cpu().numpy() * max_val
            mask_np = mask.cpu().numpy()
            
            all_preds.append(pred_denorm)
            all_trues.append(y_denorm)
            all_masks.append(mask_np)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1} 个 batch")
    
    # 拼接所有批次的结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    print(f"✅ 推理完成！结果形状: {all_preds.shape}")
    return all_preds, all_trues, all_masks


# ========================================================================================
# 核心函数 3: 指标计算
# ========================================================================================
def compute_metrics(preds, trues, masks, metric_types=['mae', 'rmse', 'wape']):
    """
    计算预测误差指标
    
    Args:
        preds: 预测值 [样本数, 节点数, 预测步数]
        trues: 真实值 [样本数, 节点数, 预测步数]
        masks: 有效掩码 [样本数, 节点数, 预测步数]
        metric_types: 要计算的指标类型
        
    Returns:
        metrics: 字典，包含各项指标值
    """
    # 获取有效的预测结果（mask > 0.5 表示有效）
    valid_idx = masks > 0.5
    
    if np.sum(valid_idx) == 0:
        print("⚠️ 警告: 没有有效的预测数据！")
        return {metric: 0.0 for metric in metric_types}
    
    pred_valid = preds[valid_idx]
    true_valid = trues[valid_idx]
    
    metrics = {}
    
    # MAE (Mean Absolute Error)
    if 'mae' in metric_types:
        metrics['mae'] = float(np.mean(np.abs(pred_valid - true_valid)))
    
    # RMSE (Root Mean Squared Error)
    if 'rmse' in metric_types:
        metrics['rmse'] = float(np.sqrt(np.mean((pred_valid - true_valid) ** 2)))
    
    # WAPE (Weighted Absolute Percentage Error)
    if 'wape' in metric_types:
        wape_val = np.sum(np.abs(pred_valid - true_valid)) / (np.sum(np.abs(true_valid)) + 1e-8)
        metrics['wape'] = float(wape_val)
    
    # MAPE (Mean Absolute Percentage Error)
    if 'mape' in metric_types:
        mape_val = np.mean(np.abs((pred_valid - true_valid) / (np.abs(true_valid) + 1e-8)))
        metrics['mape'] = float(mape_val)
    
    return metrics


# ========================================================================================
# 核心函数 4: 绘图
# ========================================================================================
def plot_comparison(predictions_dict, trues, masks, save_dir='experiments/verify_plots', 
                    sample_idx=0, path_idx=0, horizon=3):
    """
    绘制多个模型的预测值 vs 真实值对比图
    
    Args:
        predictions_dict: 字典 {模型名称: 预测结果数组}
        trues: 真实值 [样本数, 节点数, 预测步数]
        masks: 有效掩码
        save_dir: 图表保存目录
        sample_idx: 要绘图的样本索引
        path_idx: 要绘图的路径（节点）索引
        horizon: 预测步数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取选定样本和路径的真实值
    # trues 形状: [样本数, 预测步数, 节点数]
    true_sample = trues[sample_idx, :, path_idx]  # [预测步数]
    mask_sample = masks[sample_idx, :, path_idx]  # [预测步数]
    
    # 仅在掩码有效的时间步绘图
    valid_steps = np.where(mask_sample > 0.5)[0]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 绘制真实值 - 蓝色实线
    ax.plot(range(horizon), true_sample, 
            color='blue', linewidth=2.5, linestyle='-', 
            marker='o', markersize=8, label='Ground Truth', zorder=3)
    
    # 绘制各模型预测值
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    linestyles = ['--', '-.', ':', '-', '--', '-.']
    
    for model_idx, (model_name, preds) in enumerate(predictions_dict.items()):
        pred_sample = preds[sample_idx, :, path_idx]  # [预测步数]
        
        ax.plot(range(horizon), pred_sample,
                color=colors[model_idx % len(colors)],
                linewidth=2,
                linestyle=linestyles[model_idx % len(linestyles)],
                marker='s', markersize=6,
                label=f'{model_name} (Pred)',
                zorder=2, alpha=0.8)
    
    # 图表美化
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=13, fontfamily='Times New Roman', fontweight='bold')
    ax.set_ylabel('Traffic Flow (vehicles/min)', fontsize=13, fontfamily='Times New Roman', fontweight='bold')
    ax.set_title(f'Traffic Flow Prediction Comparison\nSample {sample_idx}, Path {path_idx}',
                 fontsize=14, fontfamily='SimSun', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.set_xticks(range(horizon))
    
    # 保存图表
    output_path = os.path.join(save_dir, f'comparison_sample{sample_idx}_path{path_idx}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")
    
    plt.close()


# ========================================================================================
# 核心函数 5: 生成时序对比图 (全测试集)
# ========================================================================================
def plot_time_series_comparison(predictions_dict, trues, masks, save_dir='experiments/verify_plots',
                                sample_idx=0, path_idx=0, num_steps=None):
    """
    绘制更长时间跨度的时序预测对比图（模拟滚动预测）
    
    Args:
        predictions_dict: {模型名称: 预测结果}
        trues: 真实值
        masks: 掩码
        save_dir: 保存目录
        sample_idx: 样本索引
        path_idx: 路径索引
        num_steps: 显示的时间步数 (如果为None，则显示所有)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取该路径的全部时间序列
    true_series = trues[:, path_idx, :].flatten()
    mask_series = masks[:, path_idx, :].flatten()
    
    if num_steps is None:
        num_steps = len(true_series)
    else:
        true_series = true_series[:num_steps]
        mask_series = mask_series[:num_steps]
    
    # 创建高分辨率时序图
    fig, ax = plt.subplots(figsize=(16, 6))
    
    time_indices = np.arange(len(true_series))
    
    # 绘制真实值 - 蓝色实线
    ax.plot(time_indices, true_series, 
            color='blue', linewidth=2.2, linestyle='-',
            label='Ground Truth', zorder=3, alpha=0.9)
    
    # 绘制预测值
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'cyan']
    linestyles = ['--', '-.', ':', '-', '--', '-.']
    
    for model_idx, (model_name, preds) in enumerate(predictions_dict.items()):
        pred_series = preds[:, path_idx, :].flatten()[:num_steps]
        
        ax.plot(time_indices, pred_series,
                color=colors[model_idx % len(colors)],
                linewidth=1.8,
                linestyle=linestyles[model_idx % len(linestyles)],
                label=model_name,
                zorder=2, alpha=0.8)
    
    # 高亮显示无效数据区域
    invalid_regions = np.where(mask_series < 0.5)[0]
    if len(invalid_regions) > 0:
        for invalid_idx in invalid_regions:
            ax.axvline(x=invalid_idx, color='gray', alpha=0.2, linestyle=':', linewidth=1)
    
    # 美化图表
    ax.set_xlabel('Time Steps', fontsize=13, fontfamily='Times New Roman', fontweight='bold')
    ax.set_ylabel('Traffic Flow (vehicles/min)', fontsize=13, fontfamily='Times New Roman', fontweight='bold')
    ax.set_title(f'Long-term Traffic Flow Prediction (Path {path_idx})',
                 fontsize=14, fontfamily='SimSun', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(fontsize=11, loc='best', ncol=2, framealpha=0.95)
    
    # 保存
    output_path = os.path.join(save_dir, f'timeseries_path{path_idx}_{num_steps}steps.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 时序对比图已保存: {output_path}")
    
    plt.close()


# ========================================================================================
# 辅助函数: 打印评估结果
# ========================================================================================
def print_metrics_table(model_results):
    """
    打印格式化的指标表格
    """
    print("\n" + "=" * 100)
    print("📊 模型性能对比表")
    print("=" * 100)
    print(f"{'模型名称':<30} {'MAE':<15} {'RMSE':<15} {'WAPE':<15}")
    print("-" * 100)
    
    for model_name, metrics in model_results.items():
        mae = metrics.get('mae', 0.0)
        rmse = metrics.get('rmse', 0.0)
        wape = metrics.get('wape', 0.0)
        print(f"{model_name:<30} {mae:<15.6f} {rmse:<15.6f} {wape:<15.6f}")
    
    print("=" * 100 + "\n")


# ========================================================================================
# 主函数
# ========================================================================================
def main():
    parser = ArgumentParser(description="模型推理与验证脚本")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model_dir', type=str, default='experiments/best_model',
                        help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='adaptive_semantic_base',
                        help='模型名称')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='用于绘图的样本索引')
    parser.add_argument('--path_idx', type=int, default=0,
                        help='用于绘图的路径索引')
    parser.add_argument('--num_time_steps', type=int, default=500,
                        help='时序图显示的时间步数')
    parser.add_argument('--adj_type', type=str, default='semantic', 
                        choices=['semantic', 'topo'],
                        help='邻接矩阵类型')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='推理批大小')
    parser.add_argument('--include_baselines', action='store_true',
                        help='是否包含基线模型进行对比')
    
    args = parser.parse_args()
    
    # ==================== 1. 加载配置 ====================
    print("\n" + "="*80)
    print("🔧 AST-GCN 模型推理与验证")
    print("="*80)
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    # ==================== 2. 加载数据 ====================
    print("\n📂 加载数据集...")
    dataset_path = config['path']['model_input_pt']
    
    full_dataset = TrafficDataset(
        dataset_path,
        window_size=config['train']['window_size'],
        horizon=config['train']['horizon'],
        adj_type=args.adj_type
    )
    num_nodes = full_dataset.adj.shape[0]
    max_val = full_dataset.max_val
    adj_matrix = torch.tensor(full_dataset.adj, dtype=torch.float32, device=device)
    
    # 按比例划分为测试集
    ratios = config['train'].get('split_ratio', [0.7, 0.1, 0.2])
    train_size = int(ratios[0] * len(full_dataset))
    val_size = int(ratios[1] * len(full_dataset))
    test_dataset = torch.utils.data.Subset(
        full_dataset,
        list(range(train_size + val_size, len(full_dataset)))
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✅ 数据加载完成！节点数: {num_nodes}, 最大值: {max_val:.2f}, 测试集大小: {len(test_dataset)}")
    
    # ==================== 3. 加载主模型 ====================
    print(f"\n🧠 加载模型: {args.model_name}")
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    
    # 根据模型名称判断类型
    if 'adaptive' in args.model_name:
        model_type = 'adaptive'
    elif 'standard' in args.model_name:
        model_type = 'standard_stgcn'
    else:
        model_type = 'vanilla'
    
    main_model = load_model(
        model_path=model_path,
        model_type=model_type,
        adj_matrix=adj_matrix,
        num_nodes=num_nodes,
        device=device
    )
    
    # ==================== 4. 推理 ====================
    print("\n🚀 执行主模型推理...")
    main_preds, main_trues, main_masks = inference(
        main_model, test_loader,
        adj_matrix=adj_matrix,
        device=device,
        max_val=max_val,
        model_type=model_type
    )
    
    # 计算主模型指标
    main_metrics = compute_metrics(main_preds, main_trues, main_masks,
                                    metric_types=['mae', 'rmse', 'wape'])
    
    print(f"\n✅ {args.model_name} 推理完成！")
    print(f"   MAE:  {main_metrics['mae']:.6f}")
    print(f"   RMSE: {main_metrics['rmse']:.6f}")
    print(f"   WAPE: {main_metrics['wape']:.6f}")
    
    # ==================== 5. 加载基线模型（可选） ====================
    predictions_dict = {args.model_name: main_preds}
    model_results = {args.model_name: main_metrics}
    
    if args.include_baselines:
        print("\n📊 加载基线模型进行对比...")
        
        baseline_models = [
            ('HA_Baseline', 'ha'),
            ('Linear_Baseline', 'linear'),
            ('LSTM_Baseline', 'lstm'),
            ('Standard_STGCN', 'standard_stgcn'),
        ]
        
        for baseline_name, baseline_type in baseline_models:
            try:
                baseline_model = load_model(
                    model_path=None,
                    model_type=baseline_type,
                    adj_matrix=adj_matrix,
                    num_nodes=num_nodes,
                    device=device
                )
                
                baseline_preds, _, _ = inference(
                    baseline_model, test_loader,
                    adj_matrix=adj_matrix,
                    device=device,
                    max_val=max_val,
                    model_type=baseline_type
                )
                
                baseline_metrics = compute_metrics(baseline_preds, main_trues, main_masks,
                                                   metric_types=['mae', 'rmse', 'wape'])
                
                predictions_dict[baseline_name] = baseline_preds
                model_results[baseline_name] = baseline_metrics
                
                print(f"  ✅ {baseline_name}")
                
            except Exception as e:
                print(f"  ⚠️ 加载 {baseline_name} 失败: {e}")
    
    # ==================== 6. 打印结果表格 ====================
    print_metrics_table(model_results)
    
    # ==================== 7. 绘图 ====================
    print("\n🎨 生成对比图表...")
    save_dir = 'experiments/verify_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制单步预测对比
    plot_comparison(
        predictions_dict, main_trues, main_masks,
        save_dir=save_dir,
        sample_idx=args.sample_idx,
        path_idx=args.path_idx,
        horizon=config['train']['horizon']
    )
    
    # 绘制时序对比
    plot_time_series_comparison(
        predictions_dict, main_preds, main_masks,
        save_dir=save_dir,
        sample_idx=args.sample_idx,
        path_idx=args.path_idx,
        num_steps=args.num_time_steps
    )
    
    # ==================== 8. 保存结果 ====================
    print("\n💾 保存详细结果...")
    results_dir = os.path.join(save_dir, '../predictions')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存预测结果
    for model_name, preds in predictions_dict.items():
        save_path = os.path.join(results_dir, f'{model_name}_predictions.npz')
        np.savez(save_path, predictions=preds, trues=main_trues, masks=main_masks)
        print(f"   ✅ {model_name} 结果已保存")
    
    # 保存指标汇总
    metrics_df_path = os.path.join(results_dir, 'metrics_summary.json')
    with open(metrics_df_path, 'w', encoding='utf-8') as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 指标汇总已保存: {metrics_df_path}")
    
    print("\n" + "="*80)
    print("✅ 验证完成！所有结果已保存。")
    print("="*80 + "\n")
    
    return model_results


if __name__ == '__main__':
    main()
