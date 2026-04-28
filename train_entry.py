"""
   python train_entry.py --model_type adaptive --adj_type semantic
   python train_entry.py --model_type vanilla --adj_type semantic
   python train_entry.py --model_type vanilla --adj_type topo
"""
import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# 确保程序能找到 src 文件夹
sys.path.append(os.getcwd())

from src.models import STGCN_LSTM_Adaptive, VanillaSTGCN, TrafficDataset
from src.training import Trainer

def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="Traffic Flow Prediction Training Entry")
    parser.add_argument('--model_type', type=str, default='adaptive', choices=['adaptive', 'vanilla'],
                        help='模型选择: adaptive (自适应改进型) 或 vanilla (基础对比型)')
    parser.add_argument('--adj_type', type=str, default='semantic', choices=['semantic', 'topo'],
                        help='邻接矩阵选择: semantic (语义) 或 topo (物理)')
    args = parser.parse_args()

    # 2. 加载配置
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 3. 数据准备
    print(f"📦 正在加载数据包: {config['path']['model_input_pt']}")
    full_dataset = TrafficDataset(
        pt_path=config['path']['model_input_pt'],
        window_size=config['train']['window_size'],
        horizon=config['train']['horizon'],
        adj_type=args.adj_type
    )

    # 划分训练/验证/测试集 (7:1:2)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'])

    # 4. 初始化模型
    num_nodes = full_dataset.adj.shape[0]
    if args.model_type == 'adaptive':
        print("核心模型：STGCN_LSTM_Adaptive (带自适应嵌入层)")
        model = STGCN_LSTM_Adaptive(
            adj=full_dataset.adj, 
            num_nodes=num_nodes, 
            horizon=config['train']['horizon']
        )
        exp_tag = "adaptive"
    else:
        print("基准模型：VanillaSTGCN (静态邻接矩阵)")
        model = VanillaSTGCN(
            num_nodes=num_nodes, 
            horizon=config['train']['horizon']
        )
        exp_tag = f"vanilla_{args.adj_type}"

    # 5. 设置实验产出目录
    save_dir = os.path.join(config['path']['exp_root'], f"{exp_tag}_run")
    os.makedirs(save_dir, exist_ok=True)

    # 6. 开始训练
    trainer = Trainer(model, config['train'], adj_matrix=full_dataset.adj)
    trainer.fit(train_loader, val_loader, save_path=save_dir)

    # 7. 测试与指标保存
    print("\n--- 训练结束，开始最终测试 ---")
    trainer.test(
        test_loader, 
        max_val=full_dataset.max_val, 
        save_dir=save_dir, 
        model_name=f"metrics_{exp_tag}"
    )

if __name__ == "__main__":
    main()