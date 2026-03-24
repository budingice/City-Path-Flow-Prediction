import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# ==========================================
# 1. 你的 STGCN 模型 (稍作修改以适应 buffer)
# ==========================================
class STGCN_Temporal(nn.Module):
    def __init__(self, adj, num_nodes, hidden_dim=64):
        super(STGCN_Temporal, self).__init__()
        
        # --- 空间组件：GCN 层 ---
        adj_numpy = np.array(adj)
        adj_numpy = adj_numpy + np.eye(num_nodes) # 自连接
        # 计算度矩阵 D^(-0.5)
        rowsum = adj_numpy.sum(1)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # 处理除0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        # 归一化邻接矩阵 D^(-0.5)AD^(-0.5)
        norm_adj = d_mat_inv_sqrt @ adj_numpy @ d_mat_inv_sqrt
        
        # 使用 register_buffer，它不参与训练但随模型移动到 GPU
        self.register_buffer('normalized_adj', torch.FloatTensor(norm_adj))
        
        self.gcn_mapper = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )

        # --- 时间组件 ---
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim)
        )

        # --- LSTM 层 ---
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        
        # --- 输出层 ---
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t_code):
        batch_size, T, N, F = x.shape
        
        # A. 空间特征
        x_reshaped = x.view(-1, N, F) 
        # 使用 buffer 里的 normalized_adj
        x_space = torch.matmul(self.normalized_adj, x_reshaped) 
        x_space = self.gcn_mapper(x_space) 
        x_space = x_space.view(batch_size, T, N, -1) 

        # B. 时间融合
        t_feat = self.time_embedding(t_code) 
        t_feat = t_feat.view(batch_size, 1, 1, -1).expand(-1, T, N, -1)
        x_combined = x_space + t_feat 

        # C. 序列建模
        x_combined = x_combined.permute(0, 2, 1, 3).reshape(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(x_combined)
        
        # D. 预测
        last_state = lstm_out[:, -1, :] 
        pred = self.out_fc(last_state) 
        
        return pred.view(batch_size, N)

# ==========================================
# 2. 设置参数与虚拟数据
# ==========================================
Num_Nodes = 50       # 50条路径
Time_Steps = 12     # 使用过去12个时刻(比如3小时)预测下一个
Hidden_Dim = 64
Batch_Size = 32
Epochs = 50          # 训练轮数

# 2.1 生成虚拟的 Jaccard 邻接矩阵 (50x50)
np.random.seed(42)
# 模拟一些重叠，大部分是0，少部分有值
mock_adj = np.random.rand(Num_Nodes, Num_Nodes)
mock_adj = (mock_adj > 0.85).astype(float) # 降低密度
mock_adj = (mock_adj + mock_adj.T) / 2 # 对称化
np.fill_diagonal(mock_adj, 1.0) # 对角线为1

# 2.2 实例化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STGCN_Temporal(mock_adj, Num_Nodes, Hidden_Dim).to(device)

# 2.3 模拟训练数据 (流量数据和时间编码)
# 流量数据 (Batch, T, N, 1)，模拟一个正弦波+噪声
def generate_mock_data(batch_size, t_steps, n_nodes):
    t_codes = torch.rand(batch_size, 1)
    
    # 1. 模拟时间基础波形 (batch, t_steps)
    time_base = torch.linspace(0, 4*np.pi, batch_size * t_steps).view(batch_size, t_steps)
    flow_base = 50 + 30 * torch.sin(time_base) 
    
    # 2. 扩展维度到 (batch, t_steps, n_nodes)
    # 先增加一个维度变成 (batch, t_steps, 1)，再扩展到 n_nodes
    flow = flow_base.unsqueeze(-1).expand(-1, -1, n_nodes).clone()
    
    # 3. 加入节点特性和噪声
    node_scale = torch.rand(n_nodes) * 2.0 + 0.5  # 每个节点的缩放系数
    flow = flow * node_scale + torch.randn(batch_size, t_steps, n_nodes) * 5
    
    # 4. Target 是最后一个时刻的流量 (Batch, N)
    targets = flow[:, -1, :].clone() 
    
    # 5. 返回结果，确保 X 是 (B, T, N, 1)
    return flow.unsqueeze(-1), t_codes, targets
# ==========================================
# 3. 训练过程
# ==========================================
criterion = nn.MSELoss() # 使用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"开始训练，设备: {device}, Epochs: {Epochs}...")
model.train()
for epoch in range(Epochs):
    # 模拟生成数据 (实际中这里是 DataLoader)
    data_x, data_t, data_y = generate_mock_data(Batch_Size, Time_Steps, Num_Nodes)
    data_x, data_t, data_y = data_x.to(device), data_t.to(device), data_y.to(device)
    
    optimizer.zero_grad()
    outputs = model(data_x, data_t)
    loss = criterion(outputs, data_y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{Epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 4. 预测与可视化
# ==========================================
model.eval()
print("\n正在生成测试集预测结果并画图...")

# 4.1 生成一段“连续”的测试数据模拟一天
Test_Steps = 96 # 模拟24小时 (每15分钟一个点)
test_flows = []
test_targets = []
test_preds = []

with torch.no_grad():
    for i in range(Test_Steps):
        # 为了展示连续性，我们让生成的数据带有时间的演变特性
        # 这里用一个小 tricks 模拟时间的连续流动
        t_val = (i / Test_Steps) # 0 -> 1 模拟一天
        t_code_test = torch.tensor([[t_val]]).to(device)
        
        # 生成基于当前时间的输入 X (这里简单模拟，实际应用中是真实的历史序列滑动窗口)
        # 我们假设输入序列也是随时间变化的
        base_flow = 100 + 50 * np.sin(2 * np.pi * t_val) # 模拟早晚高峰波形
        # 加上节点差异和噪声
        x_single = base_flow + torch.randn(1, Time_Steps, Num_Nodes, 1) * 10
        # 真实值 Target (我们在输入基础上稍作变动模拟“真实”的下一个时刻)
        y_single_true = base_flow + torch.randn(1, Num_Nodes) * 5
        
        x_single, y_single_true = x_single.to(device), y_single_true.to(device)
        
        # 预测
        pred_single = model(x_single, t_code_test)
        
        # 记录结果 (存回 CPU 并转为 numpy)
        test_targets.append(y_single_true.cpu().numpy()[0])
        test_preds.append(pred_single.cpu().numpy()[0])

# 转换为 (Test_Steps, Num_Nodes) 的 Numpy 数组
test_targets = np.array(test_targets)
test_preds = np.array(test_preds)

# 4.2 选择第 1 条路径 (Index 0) 进行可视化
path_index = 0
ground_truth = test_targets[:, path_index]
prediction = test_preds[:, path_index]

# 计算该路径的简单 MAE
path_mae = np.mean(np.abs(ground_truth - prediction))

# 4.3 画图
plt.figure(figsize=(12, 6))
plt.plot(ground_truth, label='真实流量 (Ground Truth)', color='blue', linewidth=2)
plt.plot(prediction, label='预测流量 (STGCN Prediction)', color='red', linestyle='--', linewidth=2)

plt.title(f'路径 {path_index} 流量预测对比图\n(模拟测试集, MAE: {path_mae:.2f})', fontsize=14)
plt.xlabel('时间步 (每15分钟)', fontsize=12)
plt.ylabel('车辆数 (Flow)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='-', alpha=0.5)

# 模拟横坐标时间 (00:00 - 23:45)
ticks = np.arange(0, Test_Steps, 12) # 每3小时标一个点
tick_labels = [f'{h:02d}:00' for h in range(0, 24, 3)]
plt.xticks(ticks, tick_labels)

plt.tight_layout()
plt.show()