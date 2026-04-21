"""
   希望参数保存为 model_results/metrics_adaptive.txt
    
"""
class STGCN_LSTM_Adaptive(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64, horizon=3):
        super(STGCN_LSTM_Adaptive, self).__init__()
        self.horizon = horizon
        self.num_nodes = num_nodes
        
        # 1. 静态部分：保留你的 Jaccard 矩阵作为先验 (Prior)
        adj = torch.FloatTensor(adj)
        adj = adj + torch.eye(num_nodes)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.static_adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        
        # 2. 自适应部分：增加两个可学习的节点嵌入向量 (Node Embeddings)
        # 通过两个向量相乘生成一个全局自适应矩阵
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(torch.float32), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(torch.float32), requires_grad=True)
        
        self.gcn = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        batch_size, T, N, F = x.shape
        
        # 计算自适应邻接矩阵: A_adapt = Softmax(Relu(E1 * E2))
        # 这捕捉了数据中潜在的动态关联
        adp = torch.mm(self.nodevec1, self.nodevec2)
        adp = torch.relu(adp)
        adp = torch.softmax(adp, dim=1) # 归一化
        
        # 融合矩阵：你可以调整静态和自适应的比例，或者直接让模型学习
        # 这里演示直接相加融合
        total_adj = self.static_adj + adp
        
        # 空间特征提取 (GCN)
        x = x.view(-1, N, F)
        x = torch.matmul(total_adj, x) # 使用融合后的矩阵
        x = self.gcn(x)
        
        # 时间特征提取 (LSTM)
        x = x.view(batch_size, T, N, -1).permute(0, 2, 1, 3)
        x = x.reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] 
        
        x = self.out_fc(x)
        x = x.view(batch_size, N, self.horizon).permute(0, 2, 1)
        return x