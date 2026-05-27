#!/usr/bin/env python
"""
========================================================================================
简化版推理脚本 - 单模型推理示例
========================================================================================

这个脚本演示如何快速对一个模型进行推理和评估。

使用方法:
  python scripts/infer_single_model.py
  
可选参数:
  --model_name   模型名称 (默认: adaptive_semantic_base)
  --model_path   模型权重路径 (默认: experiments/best_model/best_model.pth)
  --adj_type     邻接矩阵类型 (默认: semantic, 可选: topo)
========================================================================================
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import STGCN_LSTM_Adaptive, TrafficDataset
from torch.utils.data import DataLoader


def infer_and_evaluate(model, test_loader, adj_matrix, device, max_val, model_name):
    """
    对模型进行推理并计算评估指标
    """
    print(f"\n🚀 对 {model_name} 进行推理...")
    
    all_preds = []
    all_trues = []
    all_masks = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y, mask) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # 推理
            pred = model(x, adj_matrix)
            
            # 反归一化
            pred_denorm = pred.cpu().numpy() * max_val
            y_denorm = y.cpu().numpy() * max_val
            mask_np = mask.cpu().numpy()
            
            all_preds.append(pred_denorm)
            all_trues.append(y_denorm)
            all_masks.append(mask_np)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1} 个 batch")
    
    # 拼接结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    print(f"✅ 推理完成！结果形状: {all_preds.shape}")
    
    # 计算指标
    valid_idx = all_masks > 0.5
    pred_valid = all_preds[valid_idx]
    true_valid = all_trues[valid_idx]
    
    mae = float(np.mean(np.abs(pred_valid - true_valid)))
    rmse = float(np.sqrt(np.mean((pred_valid - true_valid) ** 2)))
    wape = float(np.sum(np.abs(pred_valid - true_valid)) / (np.sum(np.abs(true_valid)) + 1e-8))
    
    print(f"\n📊 评估指标:")
    print(f"  MAE:  {mae:.6f} 辆/分钟")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  WAPE: {wape:.6f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'wape': wape,
        'preds': all_preds,
        'trues': all_trues,
        'masks': all_masks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='adaptive_semantic_base')
    parser.add_argument('--model_path', type=str, default='experiments/best_model/best_model.pth')
    parser.add_argument('--adj_type', type=str, default='semantic')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"🧠 单模型推理脚本 - {args.model_name}")
    print("="*80)
    
    # 1. 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 2. 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 3. 加载数据
    print("\n📂 加载数据集...")
    dataset_path = config['path']['model_input_pt']
    
    dataset = TrafficDataset(
        dataset_path,
        window_size=config['train']['window_size'],
        horizon=config['train']['horizon'],
        adj_type=args.adj_type
    )
    
    num_nodes = dataset.adj.shape[0]
    max_val = dataset.max_val
    adj_matrix = torch.tensor(dataset.adj, dtype=torch.float32, device=device)
    
    # 划分为测试集
    ratios = config['train'].get('split_ratio', [0.7, 0.1, 0.2])
    train_size = int(ratios[0] * len(dataset))
    val_size = int(ratios[1] * len(dataset))
    test_indices = list(range(train_size + val_size, len(dataset)))
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✅ 数据加载成功！节点数: {num_nodes}, 最大值: {max_val:.2f}, 测试集大小: {len(test_subset)}")
    
    # 4. 加载模型
    print(f"\n🔄 加载模型: {args.model_path}")
    model = STGCN_LSTM_Adaptive(
        adj=adj_matrix.cpu().numpy(),
        num_nodes=num_nodes,
        hidden_dim=config['model']['hidden_dim'],
        horizon=config['train']['horizon']
    )
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("✅ 模型权重加载成功！")
    else:
        print(f"⚠️ 警告: 模型权重文件不存在 ({args.model_path})")
    
    model = model.to(device)
    model.eval()
    
    # 5. 推理和评估
    results = infer_and_evaluate(
        model, test_loader, adj_matrix, device, max_val, args.model_name
    )
    
    # 6. 保存结果
    print("\n💾 保存结果...")
    os.makedirs('experiments/predictions', exist_ok=True)
    
    save_path = f'experiments/predictions/{args.model_name}_results.npz'
    np.savez(
        save_path,
        preds=results['preds'],
        trues=results['trues'],
        masks=results['masks']
    )
    print(f"✅ 预测结果已保存: {save_path}")
    
    # 保存指标
    metrics_path = f'experiments/predictions/{args.model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'mae': results['mae'],
            'rmse': results['rmse'],
            'wape': results['wape'],
        }, f, indent=2)
    print(f"✅ 评估指标已保存: {metrics_path}")
    
    print("\n" + "="*80)
    print("✅ 推理完成！")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
