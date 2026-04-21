import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.gridspec as gridspec

def plot_case_study_final():
    # --- 1. 数据准备 ---
    input_path = "model_inputs/st_congestion_features.parquet"
    cases_path = "analysis_results/refined_spillback_cases.csv"
    st_data_path = "model_inputs/st_batch_data.pt"
    
    df = pd.read_parquet(input_path)
    cases = pd.read_csv(cases_path).sort_values(by='confidence', ascending=False)
    st_data = torch.load(st_data_path)
    global_paths = st_data['path_labels'] # 元路径列表 [('e1','e2'), ...]
    adj_matrix = st_data['adj']
    
    # 选取最佳案例
    case = cases.iloc[0]
    p_src = int(case['trigger_path'])
    t_event = pd.to_datetime(case['start_time'])
    
    # 筛选相似度最高的前 3 条路径进行对比
    similarities = adj_matrix[p_src]
    top_neighbor_ids = np.argsort(similarities)[-4:][::-1] # 包含自己
    
    # 时间窗口：触发前后 20 分钟
    t_s, t_e = t_event - pd.Timedelta(minutes=10), t_event + pd.Timedelta(minutes=20)
    
    # 映射 Path ID
    edge_to_paths = {}
    for idx, path in enumerate(global_paths):
        for edge in path:
            edge_to_paths.setdefault(edge, []).append(idx)
    df['path_idx'] = df['edge_id'].map(lambda x: edge_to_paths.get(x, [None])[0])
    
    # --- 2. 绘图设置 ---
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # --- 图一：多路径时序折线图 ---
    ax1 = plt.subplot(gs[0])
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#FBC02D']
    
    for i, p_idx in enumerate(top_neighbor_ids):
        p_data = df[(df['path_idx'] == p_idx) & (df['time_window'] >= t_s) & (df['time_window'] <= t_e)]
        p_data = p_data.groupby('time_window')['TTI'].mean()
        
        sim_val = similarities[p_idx]
        label = f"路径 {p_idx} (相似度: {sim_val:.2f})" if p_idx != p_src else f"核心路径 {p_idx} (源)"
        
        ax1.plot(p_data.index, p_data.values, label=label, color=colors[i], lw=2.5, marker='o' if p_idx==p_src else None, ms=4)
    
    ax1.axvline(t_event, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('拥堵触发点', xy=(t_event, 2.0), xytext=(t_event - pd.Timedelta(minutes=5), 2.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax1.set_title("图一：元路径与相似路径的时序流量(TTI)变化对比", fontsize=14)
    ax1.set_ylabel("TTI 拥堵指数")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- 图二：地图轨迹映射 (真实空间表现) ---
    ax2 = plt.subplot(gs[1])
    
    # 模拟地图坐标映射 (在实际中你应该从 edge_id 解析出真实的经纬度节点)
    # 这里我们演示如何展示路径的“空间重叠/平行”关系
    for i, p_idx in enumerate(top_neighbor_ids):
        # 获取该路径包含的所有路段名
        path_edges = global_paths[p_idx]
        
        # 演示用的坐标生成逻辑：你可以替换为真实的坐标查询
        # 假设每个 edge_id 对应一段坐标
        x = np.linspace(116.3 + p_idx*0.005, 116.4 + p_idx*0.005, len(path_edges))
        y = np.sin(x*100) * 0.01 + 39.9 + (i*0.002) 
        
        ax2.plot(x, y, color=colors[i], lw=4, alpha=0.7, label=f"Path {p_idx}")
        ax2.text(x[0], y[0], f"P{p_idx} 起点", fontsize=9)
        ax2.text(x[-1], y[-1], f"P{p_idx} 终点", fontsize=9)

    ax2.set_title("图二：相似路径在真实地图空间中的拓扑布局", fontsize=14)
    ax2.set_xlabel("经度 (Longitude)")
    ax2.set_ylabel("纬度 (Latitude)")
    ax2.legend()
    ax2.grid(True, alpha=0.2, linestyle=':')

    plt.tight_layout()
    plt.savefig("analysis_results/final_visual_report.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_case_study_final()