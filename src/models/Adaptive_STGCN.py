import torch
import torch.nn as nn

class STGCN_LSTM_Adaptive(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64, horizon=3):
        super(STGCN_LSTM_Adaptive, self).__init__()
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True) # 融合权重
        
        # 1. 静态部分：归一化处理
        adj = torch.FloatTensor(adj)
        adj = adj + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.static_adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        
        # 2. 自适应部分：节点嵌入
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
        
        self.gcn = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, _adj=None): 
        batch_size, T, N, F = x.shape
        
        # 1. 计算自适应邻接矩阵
        # 增加一个温度系数 0.1，让 softmax 的结果更“尖锐”，避免由于过平滑变成直线
        adp = torch.mm(self.nodevec1, self.nodevec2)
        adp = torch.softmax(torch.relu(adp) / 0.1, dim=1) 
        
        # 2. 动态加权融合
        w = torch.sigmoid(self.alpha)
        total_adj = w * self.static_adj + (1 - w) * adp
        
        # 3. 空间特征提取 (GCN)
        # 这里实际上做的是聚合：Z = (w*A_stat + (1-w)*A_adp) * X
        x_space = x.view(-1, N, F)
        x_space = torch.matmul(total_adj, x_space)
        x_space = self.gcn(x_space)
        
        # 4. 时间特征提取 (LSTM)
        # 将空间聚合后的特征重新整理回时间序列维度
        x_time = x_space.view(batch_size, T, N, -1).permute(0, 2, 1, 3) # [B, N, T, D]
        x_time = x_time.reshape(batch_size * N, T, -1) # [B*N, T, D]
        lstm_out, _ = self.lstm(x_time)
        
        # 取最后一个时间步的隐状态
        x_out = lstm_out[:, -1, :] 
        
        # 5. 输出层映射
        x_out = self.out_fc(x_out) # [B*N, horizon]
        return x_out.view(batch_size, N, self.horizon).permute(0, 2, 1) # [B, horizon, N]
    