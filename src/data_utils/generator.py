import numpy as np
import pandas as pd
import torch
import logging

class PathTensorGenerator:
    def __init__(self, config):
        """
        通过 config 初始化，直接读取 preprocess 下的参数
        """
        self.num_top_paths = config['preprocess'].get('num_top_paths', 50)
        self.time_step_sec = config['preprocess'].get('time_step_sec', 60)
        
        # 读取缺失规律配置 (解决直线化核心)
        dc_config = config['preprocess'].get('duty_cycle', {'active_minutes': 15, 'period_minutes': 30})
        self.active_minutes = dc_config['active_minutes']
        self.period_minutes = dc_config['period_minutes']
        
        self.global_paths = None
        self.path_to_idx = None

    def build_global_path_library(self, path_files):
        """筛选全局高频路径作为图节点"""
        logging.info(f"🔍 扫描全局 Top {self.num_top_paths} 路径...")
        all_path_series = []
        for f in path_files:
            temp_df = pd.read_parquet(f)
            temp_df['path_tuple'] = temp_df['path_sequence'].apply(tuple)
            all_path_series.append(temp_df['path_tuple'])
        
        self.global_paths = pd.concat(all_path_series).value_counts().head(self.num_top_paths).index.tolist()
        self.path_to_idx = {path: i for i, path in enumerate(self.global_paths)}
        return self.global_paths

    def build_semantic_adj(self):
        """构建 Jaccard 语义邻接矩阵 (A_sem)"""
        num_nodes = len(self.global_paths)
        adj = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            set_i = set(self.global_paths[i])
            for j in range(num_nodes):
                set_j = set(self.global_paths[j])
                sim = len(set_i & set_j) / len(set_i | set_j) if len(set_i | set_j) > 0 else 0
                adj[i, j] = sim
        return adj
    
    def build_topology_adj(self):
        """构建物理拓扑邻接矩阵 (A_topo)"""
        num_nodes = len(self.global_paths)
        adj = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            path_i = self.global_paths[i]
            last_edge_i = path_i[-1]
            for j in range(num_nodes):
                path_j = self.global_paths[j]
                first_edge_j = path_j[0]
                if last_edge_i == first_edge_j:
                    adj[i, j] = 1
        return adj

    def _is_active_sampling(self, timestamp):
        """
        判断当前时间戳是否在有效采样窗口内
        基于 config 里的 duty_cycle 逻辑
        """
        # 计算相对于周期的偏移（分钟）
        # 例如 pNEUMA 是每 30 分钟一个周期
        minute_in_period = (timestamp.minute + timestamp.hour * 60) % self.period_minutes
        return minute_in_period < self.active_minutes

    def generate_chunk(self, df):
        """
        将单个文件转为 [Time, Nodes, 2] 张量
        通道 0: Flow (计数)
        通道 1: Mask (0/1 掩码)
        """
        df['path_tuple'] = df['path_sequence'].apply(tuple)
        start_t = df['start_time'].min()
        end_t = df['end_time'].max()
        
        num_steps = int(np.ceil((end_t - start_t).total_seconds() / self.time_step_sec))
        
        # 初始化双通道 [T, N, 2]
        chunk = np.zeros((num_steps, self.num_top_paths, 2))
        
        # 1. 填充 Mask 通道
        for t in range(num_steps):
            current_dt = start_t + pd.Timedelta(seconds=t * self.time_step_sec)
            if self._is_active_sampling(current_dt):
                chunk[t, :, 1] = 1.0 # 标记该时间片所有节点观测有效
        
        # 2. 填充流量通道
        for _, row in df.iterrows():
            if row['path_tuple'] in self.path_to_idx:
                t_idx = int((row['start_time'] - start_t).total_seconds() // self.time_step_sec)
                p_idx = self.path_to_idx[row['path_tuple']]
                if 0 <= t_idx < num_steps:
                    chunk[t_idx, p_idx, 0] += 1
                    
        return chunk