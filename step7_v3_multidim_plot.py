import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

# 设置画图风格
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

def plot_advanced_spillback_analysis():
    # --- 1. 数据加载 ---
    input_path = "model_inputs/st_congestion_features.parquet"
    cases_path = "analysis_results/refined_spillback_cases.csv"
    st_data_path = "model_inputs/st_batch_data.pt"
    
    df = pd.read_parquet(input_path)
    cases = pd.read_csv(cases_path)
    st_data = torch.load(st_data_path)
    global_paths = st_data['path_labels']
    
    # 建立映射
    edge_to_paths = {}
    for p_idx, path_tuple in enumerate(global_paths):
        for edge in path_tuple:
            edge_to_paths.setdefault(edge, []).append(p_idx)

    # 聚合路径数据
    df['related_paths'] = df['edge_id'].map(lambda x: edge_to_paths.get(x, []))
    df_exploded = df.explode('related_paths').dropna(subset=['related_paths'])
    df_exploded['path_idx'] = df_exploded['related_paths'].astype(int)

    # --- 2. 选择一个最具代表性的时间片段 ---
    # 挑选 Confidence 最高且涉及路径最多的触发时刻
    best_event = cases.sort_values(by='confidence', ascending=False).iloc[0]
    t_event = pd.to_datetime(best_event['start_time'])
    t_window_s = t_event - pd.Timedelta(minutes=10)
    t_window_e = t_event + pd.Timedelta(minutes=20)
    
    # 提取该时段矩阵
    mask = (df_exploded['time_window'] >= t_window_s) & (df_exploded['time_window'] <= t_window_e)
    sub_df = df_exploded[mask]
    pivot_tti = sub_df.pivot_table(index='path_idx', columns='time_window', values='TTI', aggfunc='mean').fillna(1.0)

    # --- 3. 绘图 ---
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # A. 绘制背景热力图 (流量/拥堵强度)
    sns.heatmap(pivot_tti, cmap="YlOrRd", vmin=1.0, vmax=2.5, alpha=0.8,
                cbar_kws={'label': 'TTI (拥堵强度)', 'shrink': 0.8}, ax=ax)

    # B. 叠加相似度气泡
    # 筛选在该时间窗口内发生的演化案例
    current_cases = cases[(pd.to_datetime(cases['start_time']) >= t_window_s) & 
                          (pd.to_datetime(cases['start_time']) <= t_window_e)]
    
    # 时间轴坐标映射映射
    time_cols = list(pivot_tti.columns)
    path_indices = list(pivot_tti.index)

    for _, row in current_cases.iterrows():
        p_src = int(row['trigger_path'])
        p_aff = int(row['affected_path'])
        conf = row['confidence']
        t_src = pd.to_datetime(row['start_time'])
        t_aff = pd.to_datetime(row['impact_time'])

        if p_src in path_indices and t_src in time_cols:
            x_idx = time_cols.index(t_src) + 0.5
            y_idx = path_indices.index(p_src) + 0.5
            # 绘制触发点 (星号)
            ax.scatter(x_idx, y_idx, marker='*', s=200, color='blue', edgecolors='white', label='Source' if _==0 else "")
            
        if p_aff in path_indices and t_aff in time_cols:
            x_aff_idx = time_cols.index(t_aff) + 0.5
            y_aff_idx = path_indices.index(p_aff) + 0.5
            # 绘制受影响点 (气泡大小由 Confidence 决定)
            ax.scatter(x_aff_idx, y_aff_idx, s=conf*1000, color='cyan', alpha=0.6, 
                       edgecolors='black', label='Affected (Size $\propto$ Similarity)' if _==0 else "")

    # --- 4. 美化 ---
    time_labels = [t.strftime('%H:%M') for t in pivot_tti.columns]
    ax.set_xticks(np.arange(len(time_labels)) + 0.5)
    ax.set_xticklabels(time_labels, rotation=45)
    
    plt.title(f"多维关联分析: 路径演化、流量强度与空间相似度\n触发时间: {t_event}", fontsize=16)
    plt.xlabel("时间窗口", fontsize=12)
    plt.ylabel("路径索引 (Path Index)", fontsize=12)
    
    # 防止图例重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig("analysis_results/advanced_spillback_v3.png", dpi=300)
    print("✨ 方案一图表已生成: analysis_results/advanced_spillback_v3.png")
    plt.show()

if __name__ == "__main__":
    plot_advanced_spillback_analysis()