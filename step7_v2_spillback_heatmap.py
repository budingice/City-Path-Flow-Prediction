import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def plot_best_spillback_case():
    # --- 1. 路径加载 ---
    input_path = "model_inputs/st_congestion_features.parquet"
    cases_path = "analysis_results/refined_spillback_cases.csv" # 确保是刚才你贴出的内容
    st_data_path = "model_inputs/st_batch_data.pt"
    output_dir = "analysis_results"

    # --- 2. 核心映射逻辑 ---
    st_data = torch.load(st_data_path)
    global_paths = st_data['path_labels'] # List of tuples
    
    # 构建路段到路径索引的映射 (处理一多对应关系)
    edge_to_paths = {}
    for p_idx, path_tuple in enumerate(global_paths):
        for edge in path_tuple:
            edge_to_paths.setdefault(edge, []).append(p_idx)

    print(f"🔄 正在读取特征文件并聚合路径状态...")
    df = pd.read_parquet(input_path)
    
    # 筛选属于 50 条路径的路段并标记路径索引
    def map_edges_to_paths(row):
        return edge_to_paths.get(row['edge_id'], [])

    df['related_paths'] = df.apply(map_edges_to_paths, axis=1)
    df_exploded = df.explode('related_paths').dropna(subset=['related_paths'])
    df_exploded['path_idx'] = df_exploded['related_paths'].astype(int)

    # --- 3. 挑选 CSV 中的典型案例 ---
    # 我们选 Confidence 最高的案例作为展示
    cases = pd.read_csv(cases_path)
    best_case = cases.sort_values(by='confidence', ascending=False).iloc[0]
    
    p_trigger = int(best_case['trigger_path'])
    t_event = pd.to_datetime(best_case['start_time'])
    
    print(f"🎯 选定最佳展示案例：Path {p_trigger} 在 {t_event} 触发的拥堵")

    # 设定时间窗口：触发前后 15 分钟
    t_start, t_end = t_event - pd.Timedelta(minutes=15), t_event + pd.Timedelta(minutes=15)
    
    # 提取该时段内所有路径的 TTI 均值
    sub_df = df_exploded[(df_exploded['time_window'] >= t_start) & (df_exploded['time_window'] <= t_end)]
    
    # 聚合：Path + Time -> TTI Mean
    path_tti_matrix = sub_df.pivot_table(index='path_idx', columns='time_window', values='TTI', aggfunc='mean').fillna(1.0)

    # --- 4. 绘图 ---
    plt.figure(figsize=(15, 9))
    
    # 仅展示前 50 条路径中在该时段有数据的部分
    sns.heatmap(path_tti_matrix, cmap="YlOrRd", vmin=1.0, vmax=2.5, 
                cbar_kws={'label': 'TTI (1.0=Free Flow, >1.5=Congested)'})

    # 时间轴格式化
    time_ticks = [t.strftime('%H:%M') for t in path_tti_matrix.columns]
    plt.xticks(np.arange(len(time_ticks)) + 0.5, time_ticks, rotation=45)
    
    # 标注触发路径和受影响路径
    # 找出受 p_trigger 影响最明显的路径 (来自你的 CSV 数据)
    affected_nodes = cases[cases['start_time'] == best_case['start_time']]['affected_path'].unique()
    
    plt.title(f"Congestion Propagation Analysis\nTrigger: Path {p_trigger} at {t_event.strftime('%Y-%m-%d %H:%M')}", fontsize=15)
    plt.ylabel("Path Index (0-49)")
    plt.xlabel("Time Window")

    # 在纵轴上高亮 Source Path
    y_labels = path_tti_matrix.index.tolist()
    if p_trigger in y_labels:
        idx = y_labels.index(p_trigger)
        plt.gca().get_yticklabels()[idx].set_color('blue')
        plt.gca().get_yticklabels()[idx].set_weight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "congestion_propagation_final.png"), dpi=300)
    print(f"✨ 成功！请查看: {output_dir}/congestion_propagation_final.png")
    plt.show()

if __name__ == "__main__":
    plot_best_spillback_case()