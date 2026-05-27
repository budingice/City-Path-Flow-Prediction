import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from torch.utils.data import DataLoader
# 请确保你的脚本能引用到这些类，如果路径不对请修改 import 路径
from src.models.Adaptive_STGCN import STGCN_LSTM_Adaptive
from src.models.data_loader import TrafficDataset 

def run_inference():
    # 1. 自动路径定位
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "configs", "config.yaml")
    data_path = os.path.join(base_path, "data", "model_input", "st_batch_data.pt")
    model_path = os.path.join(base_path, "experiments", "adaptive_semantic_run", "best_model.pth")
    save_dir = os.path.join(base_path, "experiments", "Thesis_Plots")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载配置 (显式指定 utf-8 解决 GBK 报错)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 准备数据集 (适配你的 TrafficDataset 定义)
    print("📦 正在加载数据并进行滑窗处理...")
    dataset = TrafficDataset(
        pt_path=data_path,
        window_size=config['train']['window_size'],
        horizon=config['train']['horizon'],
        adj_type='semantic'
    )
    # 推理时 batch_size 可以设大一点提高速度
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    max_val = dataset.max_val

    # 4. 初始化模型 (适配你的 STGCN_LSTM_Adaptive 定义)
    # 你的定义: __init__(self, adj, num_nodes, hidden_dim=64, horizon=3)
    print("🧠 正在构建模型并加载权重...")
    model = STGCN_LSTM_Adaptive(
        adj=dataset.adj, 
        num_nodes=dataset.adj.shape[0],
        hidden_dim=config['model']['hidden_dim'],
        horizon=config['train']['horizon']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. 执行推理
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # 你的 forward 接收 x 和可选的 _adj
            pred = model(x) 
            
            # 反归一化还原真实物理数值
            all_preds.append(pred.cpu().numpy() * max_val)
            all_trues.append(y.cpu().numpy() * max_val)

    # 拼接结果 [Total_Samples, Horizon, Nodes]
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    # 6. 可视化：
    # 全量时间序列对比 (以第 10 条路径为例)
    path_idx = 35
    plt.figure(figsize=(18, 6), dpi=300)
    
    # 取每一个滑窗预测的第一个步长 (Horizon 0) 连成线
    plt.plot(trues[:, 0, path_idx], label='Ground Truth', color='black', alpha=0.5, linewidth=1.5)
    plt.plot(preds[:, 0, path_idx], label='Adaptive STGCN Prediction', color='#1f77b4', linestyle='--', linewidth=1.5)
    
    plt.title(f"Full-Scale Traffic Flow Inference (Path ID: {path_idx})", fontsize=14)
    plt.xlabel("Time Steps (Sequential)", fontsize=12)
    plt.ylabel("Traffic Flow Value", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_img = os.path.join(save_dir, "full_inference_result.png")
    plt.savefig(out_img, bbox_inches='tight')
    plt.show()
    
    # 7. 输出简单的性能统计
    mae = np.mean(np.abs(trues - preds))
    print(f"\n✨ 推理任务完成！")
    print(f"📊 全量数据 MAE: {mae:.4f}")
    print(f"🎨 结果图已保存至: {out_img}")

if __name__ == "__main__":
    run_inference()