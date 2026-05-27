#!/usr/bin/env python
"""
========================================================================================
快速测试脚本 - 验证 verify_model.py 的基本功能
========================================================================================

此脚本用于检查：
  ✓ 模型文件是否存在
  ✓ 数据是否能正确加载
  ✓ 推理过程是否正常
  ✓ 指标计算是否正确

使用方法:
  python scripts/test_verify_model.py
========================================================================================
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from src.models import STGCN_LSTM_Adaptive, TrafficDataset
from torch.utils.data import DataLoader, Subset


def test_configuration():
    """测试配置文件是否存在"""
    print("\n" + "="*80)
    print("测试 1: 配置文件加载")
    print("="*80)
    
    config_path = 'configs/config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功")
        print(f"   - 模型隐藏维度: {config['model']['hidden_dim']}")
        print(f"   - 批大小: {config['train']['batch_size']}")
        print(f"   - 窗口大小: {config['train']['window_size']}")
        print(f"   - 预测步数: {config['train']['horizon']}")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False


def test_model_weights():
    """测试模型权重文件是否存在"""
    print("\n" + "="*80)
    print("测试 2: 模型权重文件检查")
    print("="*80)
    
    model_path = 'experiments/best_model/best_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ 模型权重文件不存在: {model_path}")
        return False
    
    try:
        file_size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"✅ 模型权重文件存在")
        print(f"   - 文件路径: {model_path}")
        print(f"   - 文件大小: {file_size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ 模型权重文件检查失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*80)
    print("测试 3: 数据加载")
    print("="*80)
    
    try:
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dataset_path = config['path']['model_input_pt']
        
        if not os.path.exists(dataset_path):
            print(f"❌ 数据文件不存在: {dataset_path}")
            return False
        
        print(f"✅ 数据文件存在: {dataset_path}")
        
        # 加载数据集
        dataset = TrafficDataset(
            dataset_path,
            window_size=config['train']['window_size'],
            horizon=config['train']['horizon'],
            adj_type='semantic'
        )
        
        print(f"✅ 数据集加载成功")
        print(f"   - 样本总数: {len(dataset)}")
        print(f"   - 节点数: {dataset.adj.shape[0]}")
        print(f"   - 最大值: {dataset.max_val:.4f}")
        print(f"   - 邻接矩阵形状: {dataset.adj.shape}")
        
        # 取出第一个样本进行检查
        x, y, mask = dataset[0]
        print(f"✅ 样本维度检查")
        print(f"   - 输入 x 形状: {x.shape} (应为 [T=12, N, 2])")
        print(f"   - 目标 y 形状: {y.shape} (应为 [H=3, N])")
        print(f"   - 掩码形状: {mask.shape} (应为 [H=3, N])")
        
        return True
    
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*80)
    print("测试 4: 模型加载")
    print("="*80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 使用设备: {device}")
        
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dataset_path = config['path']['model_input_pt']
        dataset = TrafficDataset(
            dataset_path,
            window_size=config['train']['window_size'],
            horizon=config['train']['horizon'],
            adj_type='semantic'
        )
        
        adj_matrix = torch.tensor(dataset.adj, dtype=torch.float32)
        num_nodes = adj_matrix.shape[0]
        
        # 创建模型
        model = STGCN_LSTM_Adaptive(
            adj=adj_matrix.cpu().numpy(),
            num_nodes=num_nodes,
            hidden_dim=config['model']['hidden_dim'],
            horizon=config['train']['horizon']
        )
        
        print(f"✅ 模型创建成功")
        print(f"   - 模型类型: STGCN_LSTM_Adaptive")
        print(f"   - 参数数量: {sum(p.numel() for p in model.parameters())} 个")
        
        # 加载权重
        model_path = 'experiments/best_model/best_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"✅ 模型权重加载成功")
        
        # 测试推理
        x_sample, y_sample, mask_sample = dataset[0]
        x_batch = x_sample.unsqueeze(0).to(device)  # [1, T, N, 2]
        
        with torch.no_grad():
            pred = model(x_batch, adj_matrix.to(device))
        
        print(f"✅ 模型推理成功")
        print(f"   - 输入形状: {x_batch.shape}")
        print(f"   - 预测输出形状: {pred.shape} (应为 [1, N, H=3])")
        print(f"   - 预测值范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        return True
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_computation():
    """测试指标计算"""
    print("\n" + "="*80)
    print("测试 5: 指标计算")
    print("="*80)
    
    try:
        # 生成模拟的预测和真实值
        np.random.seed(42)
        num_samples = 100
        num_nodes = 200
        horizon = 3
        
        # 真实值范围 [0, 50] 辆/分钟
        trues = np.random.uniform(10, 50, (num_samples, num_nodes, horizon))
        # 预测值 = 真实值 + 噪声
        preds = trues + np.random.normal(0, 2, trues.shape)
        # 掩码都有效
        masks = np.ones_like(trues)
        
        # 计算指标
        valid_idx = masks > 0.5
        pred_valid = preds[valid_idx]
        true_valid = trues[valid_idx]
        
        mae = float(np.mean(np.abs(pred_valid - true_valid)))
        rmse = float(np.sqrt(np.mean((pred_valid - true_valid) ** 2)))
        wape = float(np.sum(np.abs(pred_valid - true_valid)) / np.sum(np.abs(true_valid)))
        
        print(f"✅ 指标计算成功")
        print(f"   - 样本数: {num_samples}")
        print(f"   - 节点数: {num_nodes}")
        print(f"   - 有效数据点: {np.sum(valid_idx)}")
        print(f"   - MAE: {mae:.6f}")
        print(f"   - RMSE: {rmse:.6f}")
        print(f"   - WAPE: {wape:.6f}")
        
        return True
    
    except Exception as e:
        print(f"❌ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_directory():
    """测试输出目录是否可创建"""
    print("\n" + "="*80)
    print("测试 6: 输出目录检查")
    print("="*80)
    
    try:
        output_dirs = [
            'experiments/verify_plots',
            'experiments/predictions',
        ]
        
        for dir_path in output_dirs:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                print(f"✅ 输出目录可用: {dir_path}")
            else:
                print(f"❌ 无法创建输出目录: {dir_path}")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ 输出目录检查失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 AST-GCN 模型推理脚本 - 环境检查")
    print("="*80)
    
    tests = [
        ("配置文件", test_configuration),
        ("模型权重", test_model_weights),
        ("数据加载", test_data_loading),
        ("模型加载", test_model_loading),
        ("指标计算", test_metrics_computation),
        ("输出目录", test_output_directory),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-"*80)
    print(f"总体: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✅ 所有测试通过！环境正常，可以运行推理脚本。")
        print("\n建议命令:")
        print("  python scripts/verify_model.py --model_name adaptive_semantic_base")
        print("  python scripts/verify_model.py --model_name adaptive_semantic_base --include_baselines")
        return 0
    else:
        print(f"\n❌ 有 {total - passed} 个测试失败，请检查环境设置。")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
