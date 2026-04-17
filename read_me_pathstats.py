import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

# --- 环境配置 ---
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def run_integrated_analysis(
    processed_dir='processed_data', 
    matched_dir='matched_data', 
    graph_file='athens_road_network.graphml',
    save_folder='analysis_results'
):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"📁 已创建结果目录: {save_folder}")

    # 1. 加载数据
    info_files = glob.glob(os.path.join(processed_dir, "*_info.parquet"))
    matched_files = sorted(glob.glob(os.path.join(matched_dir, "*_matched.parquet")))
    
    if not info_files or not matched_files:
        print("❌ 错误：未找到处理后的数据，请检查路径。")
        return

    print("开始整合分析...")
    df_info = pd.concat([pd.read_parquet(f) for f in info_files]).reset_index(drop=True)
    df_matched = pd.read_parquet(matched_files[0]) # 取第一个匹配文件作为时空样本
    df_matched['timestamp'] = pd.to_datetime(df_matched['timestamp'])
    
    # 加载路网
    G = ox.load_graphml(graph_file)

    # --- [模块 A: 车辆与速度统计] ---
    print("📊 统计车辆与速度特征...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # A1. 车型分布
    type_counts = df_info['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=axes[0], palette='viridis')
    axes[0].set_title('数据集车辆类型组成')
    
    # A2. 速度分布
    sns.histplot(df_matched['speed'], bins=50, kde=True, ax=axes[1], color='skyblue')
    axes[1].set_title('瞬时速度分布 (km/h)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '01_vehicle_speed_stats.png'), dpi=300)

    # --- [模块 B: 流量时间演化] ---
    print("📈 统计流量时间演化...")
    flow_min = df_matched.set_index('timestamp').resample('1T')['track_id'].nunique()
    plt.figure(figsize=(12, 5))
    plt.plot(flow_min.index, flow_min.values, color='teal', linewidth=2)
    plt.fill_between(flow_min.index, flow_min.values, alpha=0.2, color='teal')
    plt.title('监测时段交通流密度变化 (每分钟在线车辆数)')
    plt.xlabel('时间戳')
    plt.ylabel('车辆数')
    plt.savefig(os.path.join(save_folder, '02_traffic_flow_time.png'), dpi=300)

    # --- [模块 C: 路网拓扑可视化] ---
    print("🌐 绘制路网拓扑结构...")
    # 按节点度着色
    nc = ox.plot.get_node_colors_by_attr(G, "street_count", cmap="plasma")
    fig, ax = ox.plot_graph(G, node_color=nc, node_size=20, node_zorder=2, bgcolor='white',
                             edge_color='#cccccc', edge_linewidth=0.8, show=False, close=False)
    ax.set_title("Athens Road Network Topology", fontsize=15)
    plt.savefig(os.path.join(save_folder, '03_network_topology.png'), dpi=300)

    # --- [模块 D: 路段负载分析 (长尾分布)] ---
    print("🛣️ 统计路段负载情况...")
    edge_counts = df_matched['edge_id'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(edge_counts)), edge_counts.values, color='red')
    plt.title('路段负载分布 (识别核心路段)')
    plt.xlabel('路段排名 (按流量降序)')
    plt.ylabel('轨迹匹配点数')
    plt.savefig(os.path.join(save_folder, '04_edge_load_dist.png'), dpi=300)

    # --- [模块 E: 导出综合数据报告] ---
    # 车辆概览
    df_info.groupby('type')['avg_speed'].describe().to_csv(os.path.join(save_folder, 'report_vehicle_stats.csv'))
    
    # 拓扑概览
    basic_stats = ox.basic_stats(G)
    topo_summary = pd.DataFrame({
        '指标': ['节点数', '路段数', '平均节点度', '路网环路比'],
        '数值': [G.number_of_nodes(), G.number_of_edges(), np.mean([d for n,d in G.degree()]), basic_stats.get('circuity_avg', 0)]
    })
    topo_summary.to_csv(os.path.join(save_folder, 'report_topology_summary.csv'), index=False, encoding='utf_8_sig')

    print(f"✨ 所有统计图表与报告已整合至: {os.path.abspath(save_folder)}")

if __name__ == "__main__":
    run_integrated_analysis()