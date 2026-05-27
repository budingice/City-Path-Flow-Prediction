import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, pt_path, window_size=12, horizon=3, adj_type='semantic'):
        """
        适配双通道数据: [Flow, Mask]
        """
        # 加载数据包
        data = torch.load(pt_path)
        self.x_list = data['x_list']
        
        # 1. 矩阵选择逻辑
        if adj_type == 'semantic':
            self.adj = torch.tensor(data['adj_semantic'], dtype=torch.float32)
        elif adj_type == 'topo':
            self.adj = torch.tensor(data['adj_topo'], dtype=torch.float32)
        else:
            raise ValueError(f"❌ 未知的 adj_type: {adj_type}")

        self.inputs = []
        self.targets = [] # 这里将改为存储 [horizon, num_nodes, 2] (包含流量和掩码)
        
        # 2. 归一化基准
        if 'max_val' in data:
            self.max_val = float(data['max_val'])
        else:
            all_flows = [chunk[..., 0].max() for chunk in self.x_list if chunk.size > 0]
            self.max_val = float(max(all_flows)) if all_flows else 1.0

        # 3. 滑窗处理
        for chunk in self.x_list:
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
            T = chunk_tensor.shape[0]
            if T < (window_size + horizon):
                continue
            
            for i in range(T - window_size - horizon + 1):
                # 输入: (window_size, num_nodes, 2) -> 包含 [Flow, Mask]
                self.inputs.append(chunk_tensor[i : i + window_size, :, :])
                # 目标: (horizon, num_nodes, 2) -> 保持双通道，以便在 __getitem__ 中拆分 Mask
                self.targets.append(chunk_tensor[i + window_size : i + window_size + horizon, :, :])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 使用 clone() 避免原地修改影响原始列表
        x = self.inputs[idx].clone()
        y_raw = self.targets[idx].clone()
        
        # --- 归一化逻辑 ---
        # 输入 x: 只有流量通道 (index 0) 归一化
        x[:, :, 0] = x[:, :, 0] / self.max_val
        
        # 目标 y_raw: 拆分为流量标签和掩码
        y = y_raw[:, :, 0] / self.max_val  # 流量标签归一化 [horizon, num_nodes]
        mask = y_raw[:, :, 1]              # 掩码保持 0/1 [horizon, num_nodes]
        
        # 返回 3 个值，完美匹配 Trainer 中的 for x, y, mask in loader:
        return x, y, mask