import numpy as np
import pandas as pd
import torch
import os
import glob
import logging
from tqdm import tqdm

class PathTensorGenerator:
    def __init__(self, num_top_paths=50, time_step_sec=60):
        self.num_top_paths = num_top_paths
        self.time_step_sec = time_step_sec
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
            path_i = self.global_paths[i] # 这是一个路段 ID 元组
            last_edge_i = path_i[-1]
            for j in range(num_nodes):
                path_j = self.global_paths[j]
                first_edge_j = path_j[0]
                # 如果 i 的终点是 j 的起点，或者两条路径有重叠且方向一致
                if last_edge_i == first_edge_j:
                    adj[i, j] = 1
        return adj
    

    def generate_chunk(self, df):
        """将单个文件转为 [Time, Nodes, 1] 张量"""
        df['path_tuple'] = df['path_sequence'].apply(tuple)
        start_t = df['start_time'].min()
        end_t = df['end_time'].max()
        
        num_steps = int(np.ceil((end_t - start_t).total_seconds() / self.time_step_sec))
        chunk = np.zeros((num_steps, self.num_top_paths, 1))
        
        for _, row in df.iterrows():
            if row['path_tuple'] in self.path_to_idx:
                t_idx = int((row['start_time'] - start_t).total_seconds() // self.time_step_sec)
                p_idx = self.path_to_idx[row['path_tuple']]
                if 0 <= t_idx < num_steps:
                    chunk[t_idx, p_idx, 0] += 1
        return chunk