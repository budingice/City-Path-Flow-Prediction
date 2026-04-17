import torch
from torch.utils.data import Dataset, DataLoader
import os

class TrafficDataset(Dataset):
    """
    专门针对 Pneuma 15分钟断裂片段设计的加载器
    """
    def __init__(self, pt_path, window_size=3, horizon=1):
        """
        :param pt_path: step5 生成的 .pt 文件路径
        :param window_size: 历史观测时长（比如看前 8 分钟）
        :param horizon: 预测未来时长（比如预测后 3 分钟）
        """
        data = torch.load(pt_path)
        self.x_list = data['x_list']  # 每个元素是 [15, 50, 1]
        print(f"DEBUG: 第一个数据块的形状是: {self.x_list[0].shape}")
        self.adj = torch.tensor(data['adj'], dtype=torch.float32)
        
        
        self.inputs = []
        self.targets = []
        
        # --- 核心逻辑：在每个 15 分钟片段内部进行滑动窗口切割 ---
        for chunk in self.x_list:
            # chunk 形状: [Time, Nodes, Feature], 例如 [15, 50, 1]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
            
            # T 是该片段总时间步
            T = chunk_tensor.shape[0]
            
            # 滑动窗口遍历
            for i in range(T - window_size - horizon + 1):
                # 输入：从 i 到 i + window_size
                x = chunk_tensor[i : i + window_size, :, :]
                # 标签：预测接下来的 horizon 步
                y = chunk_tensor[i + window_size : i + window_size + horizon, :, 0] # 只取流量维度
                
                self.inputs.append(x)
                self.targets.append(y)
                
        print(f"✅ Dataset 构建完成！")
        print(f"📊 原始片段数: {len(self.x_list)}")
        print(f"📦 切分后的样本总数 (Samples): {len(self.inputs)}")
        print(f"📐 输入形状: {self.inputs[0].shape} (Time, Nodes, Feat)")
        print(f"📐 标签形状: {self.targets[0].shape} (Time, Nodes)")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- 测试运行代码 ---
if __name__ == "__main__":
    # 假设你刚刚运行完 build_st_features_batch()
    pt_file = "model_inputs/st_batch_data.pt"
    
    if os.path.exists(pt_file):
        # 初始化 Dataset
        # 设置 window_size=10, horizon=3 意味着用前 10 分钟预测未来 3 分钟
        dataset = TrafficDataset(pt_file, window_size=10, horizon=3)
        
        # 初始化 DataLoader (用于训练循环)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 演示读取一个 Batch
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            print(f"\n样本 Batch {batch_idx} 读取成功:")
            print(f"训练特征 X 维度: {batch_x.shape}") # [16, 10, 50, 1] -> [B, T_in, N, F]
            print(f"预测目标 Y 维度: {batch_y.shape}") # [16, 3, 50]    -> [B, T_out, N]
            break # 仅演示第一组