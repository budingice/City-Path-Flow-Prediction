import pandas as pd
import numpy as np
import torch
import os

def refine_spillback_analysis():
    input_path = "model_inputs/st_congestion_features.parquet"
    adj_data_path = "model_inputs/st_batch_data.pt"
    output_dir = "analysis_results"
    
    # 1. 加载数据
    df = pd.read_parquet(input_path)
    st_data = torch.load(adj_data_path)
    adj_matrix = st_data['adj']
    global_paths = st_data['path_labels']
    
    # 2. 提高判定门槛 (TTI > 1.8 识别更严重的拥堵)
    df['is_congested'] = df['TTI'] > 1.8
    
    # 构建路径级 Pivot
    edge_pivot = df.pivot(index='time_window', columns='edge_id', values='is_congested').fillna(False)
    path_pivot = pd.DataFrame(index=edge_pivot.index, columns=range(len(global_paths))).fillna(False)

    for i, path_tuple in enumerate(global_paths):
        existing_edges = [e for e in path_tuple if e in edge_pivot.columns]
        if existing_edges:
            path_pivot[i] = edge_pivot[existing_edges].any(axis=1)

    # 3. 识别“状态翻转”点 (State Transition)
    # 我们只关注从“不堵”变为“堵”的那一刻，这才是演化的开始
    path_starts = path_pivot.astype(int).diff() == 1

    evolution_cases = []
    # 物理约束：Jaccard 提高到 0.2，确保空间强相关
    rows, cols = np.where(adj_matrix > 0.2) 

    print(f"🧐 正在对 {len(rows)} 组强相关路径进行【因果演化】精炼扫描...")

    for i, j in zip(rows, cols):
        if i == j: continue
        
        # 寻找 A 路径发生拥堵的时刻
        trigger_times = path_starts[path_starts[i]].index
        
        for t_a in trigger_times:
            # 扫描后续 1-5 分钟
            for n in range(1, 6):
                t_b = t_a + pd.Timedelta(minutes=n)
                if t_b in path_starts.index and path_starts.loc[t_b, j]:
                    evolution_cases.append({
                        'trigger_path': i,
                        'affected_path': j,
                        'start_time': t_a,
                        'impact_time': t_b,
                        'lag': n,
                        'confidence': adj_matrix[i, j]
                    })

    results_df = pd.DataFrame(evolution_cases)
    
    # 去重：同一对路径在短时间内只记录一次演化
    if not results_df.empty:
        results_df = results_df.drop_duplicates(subset=['trigger_path', 'affected_path'], keep='first')
        results_df.to_csv(os.path.join(output_dir, "refined_spillback_cases.csv"), index=False)
        print(f"✨ 精炼完成！提取出 {len(results_df)} 个核心演化事件。")
    else:
        print("🚩 未发现显著演化，请适当下调 TTI 阈值。")

if __name__ == "__main__":
    refine_spillback_analysis()