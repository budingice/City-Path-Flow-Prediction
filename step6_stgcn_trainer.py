import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==========================================
# 1. 数据集定义 
# ==========================================
class TrafficDataset(Dataset):
    def __init__(self, pt_path, window_size=10, horizon=3, smooth_window=3):
        data = torch.load(pt_path)
        self.x_list = data['x_list']
        print(f"DEBUG: 第一个数据块的形状是: {self.x_list[0].shape}")
        
        self.samples = []
        
        for chunk in self.x_list:
            # chunk 形状: [15, 50, 1]
            T, N, F = chunk.shape
            
            # --- 平滑处理  ---
            if smooth_window and smooth_window > 1:
                df_temp = pd.DataFrame(chunk[:, :, 0])
                smoothed = df_temp.rolling(window=smooth_window, center=True, min_periods=1).mean().values
                chunk_proc = smoothed.reshape(T, N, F)
            else:
                chunk_proc = chunk

            # --- 滑动窗口切分 ---
            # 只要 window_size + horizon <= 15 就能切出样本
            for i in range(T - window_size - horizon + 1):
                x = chunk_proc[i : i + window_size, :, :]
                y = chunk_proc[i + window_size : i + window_size + horizon, :, 0]
                self.samples.append((x, y))
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ 依然没切出样本！请检查: 片段长度 {self.x_list[0].shape[0]} 是否大于 window+horizon ({window_size}+{horizon})")

        # 归一化计算
        all_x = np.array([s[0] for s in self.samples])
        self.max_val = all_x.max() if all_x.max() > 0 else 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 返回归一化后的数据
        return torch.FloatTensor(x) / self.max_val, torch.FloatTensor(y) / self.max_val

# ==========================================
# 2. STGCN-LSTM 模型定义
# ==========================================
class STGCN_LSTM(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64, horizon=3):
        super(STGCN_LSTM, self).__init__()
        # 空间层：Jaccard 邻接矩阵处理
        self.horizon = horizon
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
        #输出层:horizon
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        batch_size, T, N, F = x.shape
        # GCN 提取空间特征
        x = x.view(-1, N, F)
        x = torch.matmul(self.adj, x)
        x = self.gcn(x) # (B*T, hidden_dim)
        
        # LSTM 提取时间特征
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3) # (B, N, T, hidden_dim)
        x = x.reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x)# lstm_out: (B*N, T, hidden_dim)
        x = lstm_out[:, -1, :] # 取最后一个时刻输出(B*N, hidden_dim)
        
        x = self.out_fc(x) # (B*N, horizon)
        x = x.view(batch_size, N, self.horizon).permute(0, 2, 1) # 最终: (B, horizon, N)
        return x

# ==========================================
# 3. 核心训练与保存函数
# ==========================================
def run_training():
    # 配置参数
    horizon = 3
    BATCH_SIZE = 4
    EPOCHS = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "model_results"
    os.makedirs(save_dir, exist_ok=True)
    total_losses = []
    
    # 数据与模型加载
    dataset = TrafficDataset("model_inputs/st_batch_data.pt")
    raw_data = torch.load("model_inputs/st_batch_data.pt")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = STGCN_LSTM(raw_data['adj'], num_nodes=50, horizon=horizon).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # --- 训练循环 ---
    model.train()
    for epoch in range(EPOCHS):
        current_epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            current_epoch_loss += loss.item()
        
        avg_loss = current_epoch_loss / len(train_loader)
        total_losses.append(avg_loss) # 记录 Loss
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # --- 预测阶段 (训练完成后再进行) ---
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        # 这里 shuffle=False 保证顺序
        eval_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for x_e, y_e in eval_loader:
            p_e = model(x_e.to(device))
            # 还原归一化数值
            all_preds.append(p_e.cpu().numpy() * dataset.max_val)
            all_trues.append(y_e.cpu().numpy() * dataset.max_val)

    y_pred = np.concatenate(all_preds) # [Samples, Horizon, Nodes]
    y_true = np.concatenate(all_trues) # [Samples, Horizon, Nodes]
    
    # 1. 保存 Loss 曲线数据
    np.save(f"{save_dir}/loss_curve.npy", np.array(total_losses))
    
    # 2. 保存预测值与真实值 (用于独立绘图脚本)
    np.savez(f"{save_dir}/prediction_data.npz", 
             y_true=y_true, 
             y_pred=y_pred,
             max_val=dataset.max_val)
    
    # 3. 保存模型权重与元数据
    torch.save(model.state_dict(), f"{save_dir}/stgcn_lstm_weights.pth")
    torch.save({'max_val': dataset.max_val, 'adj': raw_data['adj']}, f"{save_dir}/meta_info.pt")
    
    print(f"💾 所有实验数据和权重已存入 {save_dir}/ 文件夹。")

    # --- 快速可视化  ---
    plt.figure(figsize=(12, 5))
    target_step = 0 
    plt.bar(np.arange(50)-0.2, y_true[0, target_step], width=0.4, label='Actual', color='gray')
    plt.bar(np.arange(50)+0.2, y_pred[0, target_step], width=0.4, label='Pred', color='blue')
    plt.title(f"Comparison (Sample 0, Step {target_step+1})")
    plt.legend()
    plt.show()

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