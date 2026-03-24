import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==========================================
# 1. 数据集定义 (含归一化参数保留)
# ==========================================
class TrafficDataset5Min(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.x_list = data['x_list']
        self.samples = []
        for chunk in self.x_list:
            if chunk.shape[0] == 3:
                x = chunk[0:2, :, :]  # 输入前2步 (T1, T2)
                y = chunk[2, :, 0]    # 预测第3步 (T3)
                self.samples.append((x, y))
        
        all_data = np.concatenate([s[0] for s in self.samples])
        self.max_val = all_data.max() if all_data.max() > 0 else 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x) / self.max_val, torch.FloatTensor(y) / self.max_val

# ==========================================
# 2. STGCN-LSTM 模型定义
# ==========================================
class STGCN_LSTM(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64):
        super(STGCN_LSTM, self).__init__()
        # 空间层：Jaccard 邻接矩阵处理
        adj = torch.FloatTensor(adj)
        adj = adj + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        
        self.gcn = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        
        # 时间层：LSTM 捕获动态依赖
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, T, N, F = x.shape
        # GCN 提取空间特征
        x = x.view(-1, N, F)
        x = torch.matmul(self.adj, x)
        x = self.gcn(x) # (B*T, 50, 64)
        
        # LSTM 提取时间特征
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3) # (B, 50, T, 64)
        x = x.reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] # 取最后一个时刻输出
        
        x = self.out_fc(x)
        return x.view(batch_size, N)

# ==========================================
# 3. 核心训练与保存函数
# ==========================================
def run_training():
    # 配置参数
    BATCH_SIZE = 4
    EPOCHS = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "model_results"
    os.makedirs(save_dir, exist_ok=True)

    # 数据与模型加载
    dataset = TrafficDataset5Min("model_inputs/st_batch_data.pt")
    raw_data = torch.load("model_inputs/st_batch_data.pt")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = STGCN_LSTM(raw_data['adj'], num_nodes=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # --- 关键：保存模型权重与元数据 ---
    torch.save(model.state_dict(), f"{save_dir}/stgcn_lstm_weights.pth")
    torch.save({'max_val': dataset.max_val, 'adj': raw_data['adj']}, f"{save_dir}/meta_info.pt")

    # --- 关键：全量预测并保留结果 (用于绘图) ---
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for x_e, y_e in eval_loader:
            p_e = model(x_e.to(device))
            all_preds.append(p_e.cpu().numpy() * dataset.max_val)
            all_trues.append(y_e.cpu().numpy() * dataset.max_val)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    # --- 可视化代码 ---
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(50)-0.2, y_true[0], width=0.4, label='Actual', color='gray')
    plt.bar(np.arange(50)+0.2, y_pred[0], width=0.4, label='Pred', color='blue')
    plt.legend()
    plt.show()
    
    # 保存全量明细 CSV (为了后续画残差分布图)
    detailed_df = pd.DataFrame({
        'Real_Value': y_true.flatten(),
        'Pred_Value': y_pred.flatten(),
        'Residual': y_true.flatten() - y_pred.flatten()
    })
    detailed_df.to_csv(f"{save_dir}/detailed_results_STGCN_LSTM.csv", index=False)
    print(f"✅ 预测明细已保留至: {save_dir}/detailed_results_STGCN_LSTM.csv")

    return y_true, y_pred

def save_metrics(y_true, y_pred, save_dir="model_results"):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    with open(f"{save_dir}/metrics_lstm.txt", "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}")
    print(f"📊 统计指标已更新: {save_dir}/metrics_lstm.txt")
def plot_worst_paths(y_true, y_pred, num_paths=3):
    # 计算每条路径的 MAE
    path_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    worst_idx = np.argsort(path_mae)[-num_paths:] # 获取误差最大的索引

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(worst_idx):
        plt.subplot(1, num_paths, i+1)
        plt.plot(y_true[:100, idx], label='Actual', color='gray', alpha=0.6)
        plt.plot(y_pred[:100, idx], label='Predicted', color='red', linestyle='--')
        plt.title(f"Path {idx} (MAE: {path_mae[idx]:.2f})")
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    y_true, y_pred = run_training()
    save_metrics(y_true, y_pred)
    plot_worst_paths(y_true, y_pred)