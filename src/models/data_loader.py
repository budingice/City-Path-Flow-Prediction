import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, pt_path, window_size=12, horizon=3, adj_type='semantic'):
        """
        :param pt_path: .pt 文件的路径
        :param window_size: 历史滑动窗口大小
        :param horizon: 预测未来步数
        :param adj_type: 矩阵类型 'semantic' 或 'topo'
        """
        # 加载阶段 7 生成的数据包
        data = torch.load(pt_path)
        self.x_list = data['x_list']
        
        # --- 核心修复：根据 adj_type 选择矩阵 ---
        if adj_type == 'semantic':
            # 确保转换为 FloatTensor
            self.adj = torch.tensor(data['adj_semantic'], dtype=torch.float32)
        elif adj_type == 'topo':
            self.adj = torch.tensor(data['adj_topo'], dtype=torch.float32)
        else:
            raise ValueError(f"❌ 未知的 adj_type: {adj_type}。请选择 'semantic' 或 'topo'。")

        self.inputs = []
        self.targets = []
        
        # 计算全局最大值用于归一化 (保持代码健壮性)
        all_max = [chunk.max() for chunk in self.x_list if chunk.size > 0]
        self.max_val = float(max(all_max)) if all_max else 1.0

        # 滑窗处理
        for chunk in self.x_list:
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
            T = chunk_tensor.shape[0]
            if T < (window_size + horizon):
                continue
            
            for i in range(T - window_size - horizon + 1):
                # 输入: (window_size, num_nodes, 1)
                self.inputs.append(chunk_tensor[i : i + window_size, :, :])
                # 输出: (horizon, num_nodes) -> 预测流量
                self.targets.append(chunk_tensor[i + window_size : i + window_size + horizon, :, 0])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 返回归一化后的数据
        return self.inputs[idx] / self.max_val, self.targets[idx] / self.max_val