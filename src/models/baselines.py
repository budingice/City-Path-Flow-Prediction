import torch
import torch.nn as nn
import numpy as np

# 1. HA (Historical Average) - 统计学逻辑实现
class HA_Baseline:
    def __init__(self, horizon=3):
        self.horizon = horizon

    def predict(self, x):
        # x: [Batch, T, N, 2]
        avg_flow = x[..., 0].mean(dim=1, keepdim=True) # [Batch, 1, N]
        return avg_flow.repeat(1, self.horizon, 1)
    
    def eval(self): pass

# 2. Linear / ARIMA-like (线性时序模型)
class Linear_Baseline(nn.Module):
    """
    线性时序回归：模拟 ARIMA 的线性预测能力
    """
    def __init__(self, window_size, horizon, num_nodes):
        super(Linear_Baseline, self).__init__()
        self.window_size = window_size
        self.horizon = horizon
        # 针对每个节点学习一个线性映射
        self.weights = nn.Parameter(torch.randn(num_nodes, window_size, horizon))
        self.bias = nn.Parameter(torch.zeros(num_nodes, horizon))

    def forward(self, x, adj=None):
        # x: [Batch, T, N, 2] -> 只用流量通道 [Batch, T, N]
        x = x[..., 0].transpose(1, 2) # [Batch, N, T]
        # einsum 实现每个节点独立的线性回归
        out = torch.einsum('bnt,nth->bnh', x, self.weights) + self.bias
        return out.transpose(1, 2) # [Batch, horizon, N]

# 3. LSTM (纯时序深度学习)
class LSTM_Baseline(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, horizon=3):
        super(LSTM_Baseline, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, adj=None):
        batch_size, T, N, F = x.shape
        x = x.transpose(1, 2).reshape(batch_size * N, T, F)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0)) 
        return out.view(batch_size, N, -1).transpose(1, 2)

# 4. Standard STGCN (固定物理邻接)
class Standard_STGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, horizon=3):
        super(Standard_STGCN, self).__init__()
        self.start_fc = nn.Linear(2, hidden_dim)
        self.gcn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, adj):
        # adj 必须是传入的静态 Jaccard 矩阵
        batch_size, T, N, F = x.shape
        x = self.start_fc(x.view(-1, N, F))
        x = torch.matmul(adj, x) 
        x = self.gcn(x)
        x = x.view(batch_size, T, N, -1).transpose(1, 2).reshape(batch_size * N, T, -1)
        _, (h_n, _) = self.lstm(x)
        return self.out_fc(h_n.squeeze(0)).view(batch_size, N, -1).transpose(1, 2)