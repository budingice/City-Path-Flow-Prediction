

import numpy as np
import glob
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import os
import glob

def visualize():
    # --- 配置 ---
    graph_file = "athens_road_network.graphml" 
    processed_dir = "processed_data"  # 对应 step1 的输出目录
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # 1. 查找处理后的文件
    # 自动获取 processed_data 下的第一个 parquet 文件进行可视化预览
    parquet_files = glob.glob(os.path.join(processed_dir, "*.parquet"))
    
    if not parquet_files:
        print(f" 找不到处理后的数据文件。请先运行 step1_parse_pneuma.py")
        return
    
    target_file = parquet_files[0] # 取第一个文件作为示例
    print(f" 正在加载数据进行可视化: {target_file}")
    df_t = pd.read_parquet(target_file)

    # 2. 获取路网（优先从本地读取）
    if os.path.exists(graph_file):
        print(f" 正在从本地加载路网: {graph_file}...")
        G = ox.load_graphml(graph_file)
    else:
        print(" 第一次运行，正在从网络下载雅典路网...")
        # 以轨迹的中心点为基准下载路网
        avg_lat, avg_lon = df_t['lat'].mean(), df_t['lon'].mean()
        # 增加 dist 参数确保覆盖范围（例如 2000米）
        G = ox.graph_from_point((avg_lat, avg_lon), dist=2000, network_type='drive')
        ox.save_graphml(G, graph_file)
        print(f" 路网已保存至本地: {graph_file}")

    # 3. 绘图（只画前10辆车，避免渲染卡顿）
    sample_tracks = df_t['track_id'].unique()[:10]
    df_sample = df_t[df_t['track_id'].isin(sample_tracks)]
    
    print(f" 正在生成地图并叠加 {len(sample_tracks)} 条随机轨迹...")
    fig, ax = ox.plot_graph(G, show=False, close=False, edge_color='#555555', 
                            edge_linewidth=0.8, node_size=0, bgcolor='white')

    # 为不同车辆设置不同颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_tracks)))

    for tid, color in zip(sample_tracks, colors):
        track_data = df_sample[df_sample['track_id'] == tid].sort_values('timestamp')
        ax.scatter(track_data['lon'], track_data['lat'], s=5, color=color, zorder=3, alpha=0.7)

    plt.title(f"Athens Traffic Visualization (Sample Tracks)")
    plt.show()

if __name__ == "__main__":
    import numpy as np # 确保导入了 numpy
    visualize()