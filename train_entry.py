import os
import sys
import yaml
import torch
import argparse
import json
from torch.utils.data import DataLoader, random_split

# 确保能找到 src 目录
sys.path.append(os.getcwd())

from src.models import STGCN_LSTM_Adaptive, VanillaSTGCN, TrafficDataset
from src.models.baselines import HA_Baseline, Linear_Baseline, LSTM_Baseline, Standard_STGCN
from src.training import Trainer

def main():
    parser = argparse.ArgumentParser(description="Traffic Prediction Training Entry")
    parser.add_argument('--model_type', type=str, default='adaptive', 
                        choices=['adaptive', 'vanilla', 'ha', 'linear', 'lstm', 'standard_stgcn'])
    parser.add_argument('--adj_type', type=str, default='semantic', choices=['semantic', 'topo'])
    parser.add_argument('--loss_mode', type=str, default='base', choices=['base', 'trend', 'weighted'])
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    
    # --- 新增：用于自动化寻优的参数接口 ---
    parser.add_argument('--hidden_dim', type=int, default=None, help='覆盖 config 中的隐藏层维度')
    parser.add_argument('--lr', type=float, default=None, help='覆盖 config 中的学习率')
    parser.add_argument('--exp_group', type=str, default=None, help='实验组名称，用于寻优分类')
    # ----------------------------------
    args = parser.parse_args()
    args = parser.parse_args()

    # 1. 加载配置（显式指定 utf-8 解决 GBK 报错）
    if not os.path.exists(args.config):
        print(f"❌ 找不到配置文件: {args.config}")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # --- 新增：动态覆盖配置逻辑 ---
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.lr is not None:
        config['train']['lr'] = args.lr
    # ----------------------------
    
    # 2. 加载数据 (根据你的 YAML 结构修正)
    # 对应你 YAML 中的 path -> model_input_pt
    try:
        dataset_path = config['path']['model_input_pt']
    except KeyError:
        print("❌ 配置文件格式错误：请检查 path 节点下是否有 'model_input_pt' 键")
        return

    if not os.path.exists(dataset_path):
        print(f"❌ 找不到数据文件: {dataset_path}，请检查路径是否正确")
        return

    # 初始化数据集
    full_dataset = TrafficDataset(
        dataset_path, 
        window_size=config['train']['window_size'],
        horizon=config['train']['horizon'],
        adj_type=args.adj_type
    )
    num_nodes = full_dataset.adj.shape[0]
    
    # 3. 划分数据集 (使用你 YAML 中的 split_ratio: [0.7, 0.1, 0.2])
    ratios = config['train'].get('split_ratio', [0.7, 0.1, 0.2])
    train_size = int(ratios[0] * len(full_dataset))
    val_size = int(ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # --- 增加检查 ---
    print(f"📊 数据划分详情: 总样本={len(full_dataset)}, 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
    
    if val_size == 0:
        raise ValueError("❌ 错误: 验证集样本数为 0！请增加数据量或调整 configs/config.yaml 中的 split_ratio。")
    
    # ----------------
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'])

    # 4. 模型实例化
    h, z = config['model']['hidden_dim'], config['train']['horizon']
    
    if args.model_type == 'adaptive':
        model = STGCN_LSTM_Adaptive(full_dataset.adj, num_nodes, h, z)
    elif args.model_type == 'ha':
        model = HA_Baseline(horizon=z)
    elif args.model_type == 'linear':
        model = Linear_Baseline(window_size=config['train']['window_size'], horizon=z, num_nodes=num_nodes)
    elif args.model_type == 'lstm':
        model = LSTM_Baseline(num_nodes=num_nodes, hidden_dim=h, horizon=z)
    elif args.model_type == 'standard_stgcn':
        model = Standard_STGCN(num_nodes=num_nodes, hidden_dim=h, horizon=z)
    else:
        model = VanillaSTGCN(num_nodes, h, z)

    # --- 修改：实验标签与目录生成逻辑 ---
    if args.exp_group:
        # 寻优模式：标签包含具体的隐藏层维度和学习率
        h_val = config['model']['hidden_dim']
        lr_val = config['train']['lr']
        exp_tag = f"{args.exp_group}_{args.model_type}_h{h_val}_lr{lr_val}"
    else:
        # 常规模式：保持你原始的标签格式
        exp_tag = f"{args.model_type}_{args.adj_type}_{args.loss_mode}"
    # ----------------------------------
    
    save_dir = os.path.join(config['path']['exp_root'], exp_tag)
    os.makedirs(save_dir, exist_ok=True)
    
   # 6. 执行训练
    trainer = Trainer(model, config['train'], adj_matrix=full_dataset.adj, loss_mode=args.loss_mode)
    
    # --- 只有非 HA 模型才运行 fit ---
    if args.model_type != 'ha':
        trainer.fit(train_loader, val_loader, config['train']['epochs'], os.path.join(save_dir, "best_model.pth"))
    else:
        print(f"📢 跳过 {args.model_type} 的训练阶段，直接执行评估...")
    
    # 7. 测试并记录结果
    trainer.test(test_loader, full_dataset.max_val, save_dir, model_name=exp_tag)
if __name__ == "__main__":
    main()