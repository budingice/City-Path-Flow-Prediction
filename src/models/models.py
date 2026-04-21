import torch
import torch.nn as nn

class STGCN_LSTM_Base(nn.Module):
    """ 基础类，包含通用的 GCN+LSTM 逻辑 """
    def __init__(self, num_nodes, hidden_dim, horizon):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.gcn_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def _temporal_process(self, x, batch_size, T, N):
        # 提取 LSTM 时间特征的通用方法
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3) 
        x = x.reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] 
        x = self.out_fc(x)
        return x.view(batch_size, N, self.horizon).permute(0, 2, 1)
    
    # --- 模型 A: 静态 Jaccard ---
class STGCN_Static(STGCN_LSTM_Base):
    def __init__(self, adj, num_nodes, hidden_dim=64, horizon=3):
        super().__init__(num_nodes, hidden_dim, horizon)
        adj = torch.FloatTensor(adj) + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.adj = nn.Parameter(d @ adj @ d, requires_grad=False)

    def forward(self, x):
        B, T, N, F = x.shape
        x_gcn = x.view(-1, N, F)
        x_gcn = torch.matmul(self.adj, x_gcn)
        x_gcn = self.gcn_net(x_gcn)
        return self._temporal_process(x_gcn, B, T, N)
    
    # --- 模型 B: 自适应矩阵 ---
class STGCN_Adaptive(STGCN_LSTM_Base):
    def __init__(self, adj, num_nodes, hidden_dim=64, horizon=3):
        super().__init__(num_nodes, hidden_dim, horizon)
        # 初始矩阵仍作为参考
        adj = torch.FloatTensor(adj) + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.static_adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        # 自适应参数
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)

    def forward(self, x):
        B, T, N, F = x.shape
        # 生成动态矩阵
        adp = torch.softmax(torch.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        total_adj = self.static_adj + adp
        
        x_gcn = x.view(-1, N, F)
        x_gcn = torch.matmul(total_adj, x_gcn)
        x_gcn = self.gcn_net(x_gcn)
        return self._temporal_process(x_gcn, B, T, N)