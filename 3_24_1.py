import torch
import torch.nn as nn

class STGCN_Temporal(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64):
        super(STGCN_Temporal, self).__init__()
        
        # --- 1. 空间组件：GCN 层 ---
        # adj 是基于轨迹重叠度计算的 Jaccard 矩阵
        adj = torch.FloatTensor(adj)
        adj = adj + torch.eye(num_nodes) # 自连接增强
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        self.adj = nn.Parameter(d @ adj @ d, requires_grad=False)
        
        self.gcn_mapper = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )

        # --- 2. 时间组件：时间位置编码层 ---
        # 将 0-1 的时间戳转化为高维特征，弥补 15 分钟断档导致的信息缺失
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim)
        )

        # --- 3. 核心建模：LSTM 层 ---
        # 处理 (B*N, T, hidden_dim) 形状的数据
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        
        # --- 4. 输出层 ---
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t_code):
        """
        x: (Batch, T, Nodes, 1) - 流量序列数据
        t_code: (Batch, 1) - 0~1 归一化后的时间戳
        """
        batch_size, T, N, F = x.shape
        
        # A. 空间特征提取
        # 合并 B 和 T 维度进行图卷积运算
        x_reshaped = x.view(-1, N, F) # (B*T, 50, 1)
        x_space = torch.matmul(self.adj, x_reshaped) 
        x_space = self.gcn_mapper(x_space) # (B*T, 50, hidden_dim)
        x_space = x_space.view(batch_size, T, N, -1) # (B, T, 50, 64)

        # B. 时间背景融合
        # 映射时间编码并广播到所有节点和时刻
        t_feat = self.time_embedding(t_code) # (B, 64)
        t_feat = t_feat.view(batch_size, 1, 1, -1).expand(-1, T, N, -1)
        
        # 融合空间特征与时间背景（相加融合）
        x_combined = x_space + t_feat 

        # C. 序列建模
        # 针对每个节点独立运行 LSTM
        x_combined = x_combined.permute(0, 2, 1, 3).reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x_combined)
        
        # D. 预测
        # 取最后一个时间步 T2 的状态来预测 T3
        last_state = lstm_out[:, -1, :] # (B*50, 64)
        pred = self.out_fc(last_state) # (B*50, 1)
        
        return pred.view(batch_size, N) # 返回 (B, 50) 形状的路径流量预测