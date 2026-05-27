import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCN_LSTM_Adaptive(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64, horizon=3):
        # 修正：super 里的类名必须与定义的类名一致
        super(STGCN_LSTM_Adaptive, self).__init__() 
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # 1. 静态邻接矩阵 (预处理好的 Jaccard)
        adj = torch.FloatTensor(adj)
        adj = adj + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.static_adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        
        # 2. 自适应嵌入
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
        
        # 3. 核心修改：空间层输入维度从 1 映射到 2
        # 现在输入是 [Flow, Mask]，所以 in_features=2
        self.start_fc = nn.Linear(2, hidden_dim) 
        
        self.gcn_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, _adj=None):
        """
        x: [Batch, T, N, 2] -> 包含 Flow 和 Mask
        """
        batch_size, T, N, F_dim = x.shape
        
        # --- A. 空间层 (GCN) + 残差 ---
        # 首先将输入特征投影到隐藏维度 [Batch*T, N, hidden_dim]
        # x.view 之后维度变为 (Batch*T, N, 2)，刚好匹配 start_fc 的输入
        x = self.start_fc(x.view(-1, N, F_dim)) 
        residual_spatial = x  # 保存残差备份 [Batch*T, N, hidden_dim]
        
        # 计算融合矩阵 (静态 + 自适应)
        adp = torch.softmax(torch.relu(torch.mm(self.nodevec1, self.nodevec2)) / 0.1, dim=1)
        w = torch.sigmoid(self.alpha)
        total_adj = w * self.static_adj + (1 - w) * adp
        
        # 聚合空间特征
        # total_adj: [N, N], x: [Batch*T, N, hidden_dim]
        x = torch.matmul(total_adj, x)
        x = self.gcn_block(x)
        
        # 执行第一个残差连接
        x = x + residual_spatial 
        
        # --- B. 时间层 (LSTM) + 残差 ---
        # 转换形状以适应 LSTM: [Batch, N, T, hidden_dim]
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3)
        # 摊平 Batch 和 Nodes: [Batch*N, T, hidden_dim]
        x = x.reshape(batch_size * N, T, -1)
        
        residual_temporal = x[:, -1, :] # 取最后一个时间步作为残差备份
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] # [Batch*N, hidden_dim]
        
        # 执行第二个残差连接
        x = x + residual_temporal
        
        # --- C. 输出层 ---
        # x: [Batch*N, horizon]
        x = self.out_fc(x)
        
        # 最终形状还原为: [Batch, horizon, N]
        return x.view(batch_size, N, self.horizon).permute(0, 2, 1)