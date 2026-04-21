import osmnx as ox
import pandas as pd
import networkx as nx
import os
import numpy as np

def analyze_topology_only(graph_file="athens_road_network.graphml", save_folder='analysis_results'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 1. 加载路网
    if not os.path.exists(graph_file):
        print(f"❌ 找不到路网文件: {graph_file}")
        return
    
    G = ox.load_graphml(graph_file)
    print(f"🌐 成功加载路网，正在进行拓扑计算...")

    # 2. 获取核心统计指标 (basic_stats)
    # 这个函数会自动计算路网的几何与拓扑属性
    stats = ox.basic_stats(G)
    
    # 3. 区分节点类型 (基于 street_count)
    # street_count 能够排除双向车道造成的“伪度数”，准确识别物理路口
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    
    dead_ends = len(nodes_gdf[nodes_gdf['street_count'] == 1])        # 死胡同
    passing_nodes = len(nodes_gdf[nodes_gdf['street_count'] == 2])     # 道路中间点 (度数常为4)
    intersections_3way = len(nodes_gdf[nodes_gdf['street_count'] == 3]) # 三岔路口
    intersections_4way = len(nodes_gdf[nodes_gdf['street_count'] == 4]) # 十字路口
    intersections_multi = len(nodes_gdf[nodes_gdf['street_count'] >= 5]) # 多路口

    # 4. 构建汇总表格
    topology_data = {
        '拓扑指标名称': [
            '节点总数 (Nodes)', 
            '有向路段总数 (Edges)', 
            '物理街道总数 (Unique Streets)',
            '平均节点度 (Avg Degree)',
            '实际交叉口总数 (Intersections 3-way+)',
            '道路中间点数 (Passing Nodes)',
            '死胡同数量 (Dead-ends)',
            '路网环路比 (Circuity)',
            '平均路段长度 (Avg Segment Length)',
            '路网总长度 (Total Length)'
        ],
        '统计数值': [
            G.number_of_nodes(),
            G.number_of_edges(),
            stats.get('n_unique_streets'),
            round(np.mean([d for n, d in G.degree()]), 2),
            (intersections_3way + intersections_4way + intersections_multi),
            passing_nodes,
            dead_ends,
            round(stats.get('circuity_avg', 0), 4),
            f"{round(stats.get('edge_length_avg', 0), 2)} 米",
            f"{round(stats.get('edge_length_total', 0) / 1000, 2)} 公里"
        ]
    }
    # 统计有多少条唯一名称的街道
    gdf_edges = ox.graph_to_gdfs(G, nodes=False)
    if 'name' in gdf_edges.columns:
        # 处理 list 类型，使其可哈希
        clean_names = gdf_edges['name'].dropna().apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
        unique_streets = clean_names.unique()
        print(f"物理街道名称总数: {len(unique_streets)}")
        
        street_split_counts = clean_names.value_counts().sort_values(ascending=False)
        print("\n物理路拆分情况 (Top 5):")
        print(street_split_counts.head(5))
        
        # 将这个统计也加入到报告中（可选）
        print(f"平均每条物理路被拆分为: {street_split_counts.mean():.2f} 个拓扑单元")
    else:
        print("数据中不包含街道名称字段")
    # 保存为 CSV 方便在 Excel 中直接生成 PPT 表格
    df_report = pd.DataFrame(topology_data)
    report_path = os.path.join(save_folder, 'road_topology_report.csv')
    df_report.to_csv(report_path, index=False, encoding='utf_8_sig')

    # 5. 打印控制台摘要
    print("\n" + "="*40)
    print("📊 路网拓扑统计摘要 (汇报建议数据)")
    print("="*40)
    print(f"1. 空间单元数: {G.number_of_edges()} 条路段 (预测目标)")
    print(f"2. 节点规模: {G.number_of_nodes()} 个 (图神经网络节点)")
    print(f"3. 核心枢纽: {intersections_3way + intersections_4way + intersections_multi} 个真实路口")
    print(f"4. 路网特征: 平均路段长 {stats.get('edge_length_avg'):.2f}m，环路比 {stats.get('circuity_avg'):.4f}")
    print("="*40)
    print(f"✨ 完整详细报告已保存至: {report_path}")

if __name__ == "__main__":
    analyze_topology_only()