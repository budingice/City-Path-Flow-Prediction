import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaSTGCN(nn.Module):
    """
    对照组：只使用静态邻接矩阵，不含自适应嵌入
    更新：适配 [Flow, Mask] 双通道输入
    """
    def __init__(self, num_nodes, hidden_dim=64, horizon=3):
        super(VanillaSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        # 1. 核心修改：输入维度从 1 映射到 2
        self.start_fc = nn.Linear(2, hidden_dim) 
        
        # 2. 简化的 GCN 层
        self.gcn_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, adj):
        """
        x: [Batch, T, N, 2]
        adj: 外部传入的静态邻接矩阵 [N, N] (通常是 Jaccard 矩阵)
        """
        batch_size, T, N, F_dim = x.shape
        
        # --- A. 空间层 (静态 GCN) ---
        # 映射输入维度并对齐隐藏层
        x = self.start_fc(x.view(-1, N, F_dim)) # [Batch*T, N, hidden_dim]
        
        # 只使用外部传入的静态 adj 进行特征聚合
        x = torch.matmul(adj, x) 
        x = self.gcn_block(x)
        
        # --- B. 时间层 (LSTM) ---
        # 转换形状: [Batch*N, T, hidden_dim]
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3)
        x = x.reshape(batch_size * N, T, -1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] # [Batch*N, hidden_dim]
        
        # --- C. 输出层 ---
        x = self.out_fc(x) # [Batch*N, horizon]
        
        # 返回: [Batch, horizon, N]
        return x.view(batch_size, N, self.horizon).permute(0, 2, 1)