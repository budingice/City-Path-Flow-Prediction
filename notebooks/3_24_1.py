import torch
import torch.nn as nn

class STGCN_LSTM_Deep(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64, dropout=0.2):
        super(STGCN_LSTM_Deep, self).__init__()
        
        # --- 空间层：2层 GCN (邻接矩阵归一化保持一致) ---
        adj = torch.FloatTensor(adj) + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        
        # 第一层空间变换：输入 1 -> 32
        self.gcn1 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.BatchNorm1d(num_nodes) # 引入 BatchNorm 加速收敛
        )
        # 第二层空间变换：32 -> hidden_dim
        self.gcn2 = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # --- 时间融合：引入之前讨论的门控机制 ---
        self.time_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid()
        )
        self.time_offset = nn.Linear(1, hidden_dim)

        # --- 时间层：2层 LSTM ---
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2,           # 关键：堆叠 2 层
            batch_first=True,
            dropout=dropout         # 层间 Dropout
        )
        
        # 输出层
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        batch_size, T, N, F = x.shape # (B, 2, 50, 1)
        
        # 1. 空间提取 (执行两次矩阵乘法实现 2-hop 聚合)
        # 第一层 GCN
        x_s = torch.matmul(self.adj, x.view(-1, N, F)) # (B*T, N, 1)
        x_s = self.gcn1(x_s)                           # (B*T, N, 32)
        # 第二层 GCN
        x_s = torch.matmul(self.adj, x_s)              # (B*T, N, 32)
        x_s = self.gcn2(x_s).view(batch_size, T, N, -1)# (B, T, N, H)
        
        # 2. 门控融合
        gate = self.time_gate(t).view(batch_size, 1, 1, -1)
        offset = self.time_offset(t).view(batch_size, 1, 1, -1)
        x_combined = (x_s * gate) + offset
        
        # 3. 序列提取 (2 层 LSTM)
        x_combined = x_combined.permute(0, 2, 1, 3).reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x_combined) # (B*N, T, H)
        
        # 4. 预测最后一步 (未来第 3 个时间步)
        out = self.out_fc(lstm_out[:, -1, :]) # (B*N, 1)
        return out.view(batch_size, N)

# 1. 实例化模型 (假设 adj 已经准备好)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STGCN_LSTM_Deep(adj, num_nodes=50, hidden_dim=64).to(device)

# 2. 定义加权损失函数 (解决残差右偏的关键)
def peak_weighted_loss(pred, target, alpha=2.0):
    mse = (pred - target) ** 2
    # 动态权重：对高于均值的流量样本施加 alpha 倍惩罚
    weights = torch.ones_like(target)
    weights[target > target.mean()] *= alpha 
    return (mse * weights).mean()

# 3. 优化器与学习率调度
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

def train_one_epoch(model, dataloader, optimizer, device, alpha=2.0):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y, batch_t in dataloader:
        # 数据搬运到 GPU
        batch_x = batch_x.to(device) # (B, 2, 50, 1)
        batch_y = batch_y.to(device) # (B, 50)
        batch_t = batch_t.to(device) # (B, 1)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(batch_x, batch_t)
        
        # 计算加权损失
        loss = peak_weighted_loss(output, batch_y, alpha=alpha)
        
        # 反向传播
        loss.backward()
        
        # --- 梯度裁剪：防止 2 层 LSTM 梯度爆炸 ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)