import torch
import torch.nn as nn

class VanillaSTGCN(nn.Module):
    """
    对照组：只使用静态邻接矩阵，不含自适应嵌入
    """
    def __init__(self, num_nodes, hidden_dim=64, horizon=3):
        super(VanillaSTGCN, self).__init__()
        self.horizon = horizon
        self.gcn = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, adj):
        # 使用外部传入的静态矩阵 (adj)
        batch_size, T, N, F = x.shape
        x = x.view(-1, N, F)
        x = torch.matmul(adj, x)
        x = self.gcn(x)
        
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3).reshape(batch_size * N, T, -1)
        out, _ = self.lstm(x)
        return self.out_fc(out[:, -1, :]).view(batch_size, N, self.horizon).permute(0, 2, 1)