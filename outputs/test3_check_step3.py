import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import os
import glob

def visualize_step3_fixed():
    # 1. 加载路网
    graph_file = "athens_road_network.graphml"
    if not os.path.exists(graph_file): return
    G = ox.load_graphml(graph_file)

    # 2. 汇总流量
    matched_files = glob.glob(os.path.join("matched_data", "*_matched.parquet"))
    edge_counts = {}
    for f in matched_files:
        df = pd.read_parquet(f)
        counts = df['edge_id'].value_counts()
        for eid, freq in counts.items():
            edge_counts[eid] = edge_counts.get(eid, 0) + freq

    # 3. 【关键修正】：直接将属性写入 G，并提取绘图所需的属性列表
    # 我们先给所有边设置默认值，确保每一条边都有这个属性
    for u, v, k, data in G.edges(keys=True, data=True):
        eid = f"{u}_{v}"
        data['flow_count'] = edge_counts.get(eid, 0)

    # 4. 从 G 中提取流量序列，确保顺序与 OSMnx 绘图顺序完全一致
    # 这种写法保证了 list 的长度永远等于路网边的数量
    edge_flow_list = [data['flow_count'] for u, v, k, data in G.edges(keys=True, data=True)]
    max_f = max(edge_flow_list) if edge_flow_list else 1
    
    # 5. 计算颜色和线宽
    norm = colors.LogNorm(vmin=1, vmax=max_f + 1)
    cmap = plt.get_cmap('YlOrRd')
    
    ec = [cmap(norm(f)) if f > 0 else "#222222" for f in edge_flow_list]
    lw = [1.0 + (f / max_f) * 8 if f > 0 else 0.5 for f in edge_flow_list]

    print(f"🎨 正在绘制热力图，边数量: {len(ec)}")

    # 6. 绘图
    fig, ax = ox.plot_graph(
        G, node_size=0, 
        edge_color=ec, 
        edge_linewidth=lw,
        bgcolor='#111111', 
        show=False, 
        close=False
    )

    # 7. 局部缩放 (基于有流量的路段)
    active_nodes = [u for u, v, k, d in G.edges(keys=True, data=True) if d['flow_count'] > 0]
    if active_nodes:
        lats = [G.nodes[n]['y'] for n in active_nodes]
        lons = [G.nodes[n]['x'] for n in active_nodes]
        margin = 0.002
        ax.set_ylim(min(lats)-margin, max(lats)+margin)
        ax.set_xlim(min(lons)-margin, max(lons)+margin)

    # 添加颜色条
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)

    plt.title("Athens Area 1 - Final Map Matching Heatmap", color='white')
    plt.show()

if __name__ == "__main__":
    visualize_step3_fixed()