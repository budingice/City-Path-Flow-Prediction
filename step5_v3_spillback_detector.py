import pandas as pd
import numpy as np
import torch
import os

def detect_spillback_evolution():
    input_path = "model_inputs/st_congestion_features.parquet" # 包含 edge_id 的特征
    adj_data_path = "model_inputs/st_batch_data.pt"
    output_dir = "analysis_results"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. 加载数据
    df = pd.read_parquet(input_path)
    st_data = torch.load(adj_data_path)
    adj_matrix = st_data['adj']
    global_paths = st_data['path_labels'] # 这是 Tuple 组成的 List
    
    # 2. 拥堵判定
    df['is_congested'] = df['TTI'] > 1.3
    
    # --- 关键步骤：将路段级的拥堵映射到路径级 ---
    print("🔄 正在将路段级拥堵状态映射至 50 条全局路径...")
    
    # 创建一个空的映射表：Index 为时间，Columns 为 0-49 (路径索引)
    unique_times = df['time_window'].unique()
    path_pivot = pd.DataFrame(index=unique_times, columns=range(len(global_paths))).fillna(False)
    
    # 建立 edge 到 拥堵状态的时间透视，方便查询
    edge_pivot = df.pivot(index='time_window', columns='edge_id', values='is_congested').fillna(False)
    edge_pivot.columns = edge_pivot.columns.astype(str)

    for i, path_tuple in enumerate(global_paths):
        # 只要该路径中包含的任何一个路段在此时刻堵塞，我们就认为该路径整体受阻（或取均值，这里用 or 逻辑）
        existing_edges = [e for e in path_tuple if e in edge_pivot.columns]
        if existing_edges:
            # 路径 i 在 t 时刻的状态 = 其包含的所有路段状态的并集 (any)
            path_pivot[i] = edge_pivot[existing_edges].any(axis=1)

    print(f"✅ 路径级状态构建完成。有效路径列数: {path_pivot.any().sum()}")

    # 3. 演化检测
    evolution_cases = []
    lag_window = 10 
    rows, cols = np.where(adj_matrix > 0.05) # 使用 Jaccard 约束
    
    print(f"🚀 开始扫描 {len(rows)} 组路径对的时空演化...")

    for i, j in zip(rows, cols):
        if i == j: continue
        
        # 路径 A 和 路径 B 的序列
        series_A = path_pivot[i]
        series_B = path_pivot[j]
        
        if not series_A.any() or not series_B.any(): continue

        for n in range(1, lag_window + 1):
            series_B_future = series_B.shift(-n)
            spillback_mask = series_A & series_B_future
            
            if spillback_mask.any():
                event_times = spillback_mask[spillback_mask == True].index
                for t in event_times:
                    evolution_cases.append({
                        'source_path_idx': i,
                        'target_path_idx': j,
                        't_start': t,
                        'lag': n,
                        'jaccard': adj_matrix[i, j]
                    })

    if evolution_cases:
        results_df = pd.DataFrame(evolution_cases)
        results_df.to_csv(os.path.join(output_dir, "spillback_evolution_cases.csv"), index=False)
        print(f"✨ 成功！识别出 {len(results_df)} 个演化案例。")
    else:
        print("🚩 警告：即便映射后结果仍为 0。可能原因：数据时间跨度太短，不足以观察到扩散。")

if __name__ == "__main__":
    detect_spillback_evolution()