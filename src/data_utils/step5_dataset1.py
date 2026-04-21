import torch
from torch.utils.data import Dataset, DataLoader
import os

class TrafficDataset(Dataset):
    """
    针对 pNEUMA 轨迹数据片段设计的加载器
    包含：自动归一化、滑动窗口切分、多步预测标签提取
    """
    def __init__(self, pt_path, window_size=10, horizon=3):
        """
        :param pt_path: step5 生成的 .pt 文件路径
        :param window_size: 历史观测步数
        :param horizon: 预测未来步数
        """
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"❌ 找不到数据文件: {pt_path}")

        data = torch.load(pt_path)
        self.x_list = data['x_list']  # 原始列表，每个元素为 [Time, Nodes, 1]
        self.adj = torch.tensor(data['adj'], dtype=torch.float32)
        
        self.inputs = []
        self.targets = []
        
        # --- 1. 计算全局最大值用于归一化 (核心修复) ---
        all_max = []
        for chunk in self.x_list:
            all_max.append(chunk.max())
        self.max_val = max(all_max).item() if len(all_max) > 0 else 1.0
        
        # --- 2. 核心逻辑：滑动窗口切割 ---
        for chunk in self.x_list:
            # 确保 chunk 是 tensor 格式
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
            T = chunk_tensor.shape[0]
            
            # 如果片段长度不足以支撑一次窗口切分，跳过该片段
            if T < (window_size + horizon):
                continue
                
            for i in range(T - window_size - horizon + 1):
                # 输入形状: [window_size, num_nodes, 1]
                x = chunk_tensor[i : i + window_size, :, :]
                # 标签形状: [horizon, num_nodes] (只取流量维度)
                y = chunk_tensor[i + window_size : i + window_size + horizon, :, 0]
                
                self.inputs.append(x)
                self.targets.append(y)
                
        print(f"✅ Dataset 构建完成！")
        print(f"📊 最大流量值 (max_val): {self.max_val}")
        print(f"📦 样本总数: {len(self.inputs)} (窗口:{window_size}, 预测:{horizon})")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # --- 3. 归一化输出 (核心修复) ---
        # 工业界标准：模型内部处理 [0, 1] 之间的数据
        x_norm = self.inputs[idx] / self.max_val
        y_norm = self.targets[idx] / self.max_val
        return x_norm, y_norm

# 测试代码
if __name__ == "__main__":
    pt_file = "model_inputs/st_batch_data.pt"
    if os.path.exists(pt_file):
        dataset = TrafficDataset(pt_file, window_size=10, horizon=3)
        print(f"📐 样本 X 形状: {dataset[0][0].shape}") # 预期 [10, 50, 1]
        print(f"📐 样本 Y 形状: {dataset[0][1].shape}") # 预期 [3, 50]