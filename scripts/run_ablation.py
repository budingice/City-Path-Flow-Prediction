import torch
from torch.utils.data import DataLoader
from models import STGCN_Static, STGCN_Adaptive
from trainer import train_model, evaluate_and_save
from step5_dataset1 import TrafficDataset  # 确保你已经按照上一条建议修改了 TrafficDataset
import os

def main():
    # ==========================================
    # 1. 全局实验配置 (Ablation Configuration)
    # ==========================================
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'epochs': 100,
        'lr': 0.001,
        'batch_size': 16,     # 建议增加到 16 以提高训练稳定性
        'window_size': 10,    # 历史观测 10 步
        'horizon': 3          # 预测未来 3 步 (确保此处与 Dataset 一致)
    }
    
    input_pt = "model_inputs/st_batch_data.pt"
    save_dir = "model_results"
    
    if not os.path.exists(input_pt):
        print(f"❌ 找不到数据文件 {input_pt}，请先运行 step5 相关脚本。")
        return

    # ==========================================
    # 2. 数据准备
    # ==========================================
    # 这里会触发你修改后的 TrafficDataset.__init__
    dataset = TrafficDataset(input_pt, 
                            window_size=config['window_size'], 
                            horizon=config['horizon'])
    
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 加载原始数据以获取邻接矩阵
    raw_data = torch.load(input_pt)
    adj_matrix = raw_data['adj']
    num_nodes = adj_matrix.shape[0]

    # ==========================================
    # 3. 实验 A: 静态邻接矩阵 (Baseline)
    # ==========================================
    print("\n" + "="*30)
    print("🚀 启动实验 A: Static Jaccard Matrix")
    print("="*30)
    
    model_static = STGCN_Static(adj_matrix, 
                               num_nodes=num_nodes, 
                               hidden_dim=64, 
                               horizon=config['horizon'])
    
    # 开始训练
    train_model(model_static, train_loader, config)
    # 评估并保存结果
    evaluate_and_save(model_static, dataset, "Static_Model", save_dir)

    # ==========================================
    # 4. 实验 B: 自适应邻接矩阵 (Ablation)
    # ==========================================
    print("\n" + "="*30)
    print("🚀 启动实验 B: Adaptive Matrix")
    print("="*30)
    
    model_adaptive = STGCN_Adaptive(adj_matrix, 
                                   num_nodes=num_nodes, 
                                   hidden_dim=64, 
                                   horizon=config['horizon'])
    
    # 开始训练
    train_model(model_adaptive, train_loader, config)
    # 评估并保存结果
    evaluate_and_save(model_adaptive, dataset, "Adaptive_Model", save_dir)

    print("\n✨ 所有消融实验已完成！请运行 evaluator.py 查看对比图表。")

if __name__ == "__main__":
    main()