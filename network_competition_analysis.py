"""network_competition_analysis.py

网络级别的路径竞争分析 - 寻找"幽灵竞争"路径对
即：高度负相关、拓扑较独立、地理距离远的路径对

功能：
1. 时间序列提取：全路网高频路径的同步流量序列
2. 相关性挖掘：计算全路径对的 Spearman 相关矩阵，筛选 r < -0.4 的点对
3. 拓扑独立性校验：计算 Jaccard 相似度，保留 Jaccard <= max_jaccard 的路径（可调）
4. 地理距离校验：计算路径质心 Haversine 距离，保留 > min_dist km 的路径对
5. 可视化映射：在 Folium 地图上用不同颜色虚线连接"幽灵竞争"路径

用法示例:
    # 默认参数（Jaccard <= 0.1）
    python network_competition_analysis.py --min-flow 30 --min-corr -0.45 --min-dist 1.5
    
    # 严格约束：仅保留完全独立的路径（Jaccard = 0）
    python network_competition_analysis.py --max-jaccard 0.0
    
    # 宽松约束：允许30%的共享路段
    python network_competition_analysis.py --max-jaccard 0.3
    
    # 自定义所有参数
    python network_competition_analysis.py --min-flow 10 --min-corr -0.35 --min-dist 0.5 --max-jaccard 0.3
"""

import os
import glob
import pandas as pd
import numpy as np
import folium
import ast
from datetime import datetime
from scipy.spatial.distance import cdist

try:
    import networkx as nx
except ImportError:
    raise ImportError("缺少 networkx，请安装：pip install networkx")

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


# ============================================================
# 1. 数据加载工具
# ============================================================
def load_kinematics_data(input_dir="path_data"):
    """
    加载所有轨迹数据文件并合并
    返回包含 [start_time, track_id, path_signature, path_sequence] 的 DataFrame
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*_path_kinematics.parquet")))
    if not files:
        raise FileNotFoundError(f"未发现路径数据文件: {input_dir}")

    print(f"📂 加载 {len(files)} 个轨迹文件...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    print(f"✅ 共加载 {len(df)} 条轨迹记录，时间范围：{df['start_time'].min()} 至 {df['start_time'].max()}")
    return df


def load_graph_and_coords(graph_file="athens_road_network.graphml"):
    """
    加载 GraphML 路网文件并提取边坐标映射
    返回 (G, coords_dict)
    """
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"未找到路网文件: {graph_file}")
    
    print(f"🗺️ 读取路网: {graph_file}")
    if HAS_OSMNX:
        G = ox.load_graphml(graph_file)
    else:
        G = nx.read_graphml(graph_file)
    
    print(f"✅ 路网加载完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
    
    coords_dict = {}
    for u, v, key, data in (G.edges(keys=True, data=True) if isinstance(G, nx.MultiDiGraph) 
                             else [(u, v, 0, data) for u, v, data in G.edges(data=True)]):
        edge_id = f"{u}_{v}"
        
        # 从 geometry 字段提取坐标序列
        geom = data.get('geometry')
        if geom:
            if hasattr(geom, 'coords'):
                coords_dict[edge_id] = [[lat, lon] for lon, lat in geom.coords]
            elif isinstance(geom, str) and 'LINESTRING' in geom:
                coords = []
                items = geom.replace('LINESTRING(', '').replace(')', '').split(',')
                for item in items:
                    lon, lat = map(float, item.strip().split())
                    coords.append([lat, lon])
                coords_dict[edge_id] = coords
        
        # 备用：使用节点端点坐标
        if edge_id not in coords_dict:
            try:
                u_node = G.nodes[u]
                v_node = G.nodes[v]
                u_coords = [float(u_node.get('y', u_node.get('lat', 0))), 
                           float(u_node.get('x', u_node.get('lon', 0)))]
                v_coords = [float(v_node.get('y', v_node.get('lat', 0))), 
                           float(v_node.get('x', v_node.get('lon', 0)))]
                coords_dict[edge_id] = [u_coords, v_coords]
            except:
                pass
    
    print(f"✅ 提取 {len(coords_dict)} 条边的坐标数据")
    return G, coords_dict


# ============================================================
# 2. 数据预处理
# ============================================================
def parse_path_sequence(seq):
    """解析路径序列（支持多种格式）"""
    if isinstance(seq, (list, tuple, np.ndarray)):
        return list(seq)
    try:
        parsed = ast.literal_eval(str(seq))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except:
        return []


def prepare_time_series(df):
    """
    构建全路网的时间序列路径流矩阵
    行：15min 时间片，列：路径指纹，值：车辆计数
    """
    print("⏱️ 构建时间序列矩阵...")
    
    # 按 15 分钟切片
    df_ts = df.copy()
    df_ts['time_bin'] = df_ts['start_time'].dt.floor('15min')
    
    # 创建透视表：时间 × 路径
    pivot_flow = df_ts.pivot_table(
        index='time_bin',
        columns='path_signature',
        values='track_id',
        aggfunc='count'
    ).fillna(0)
    
    # 过滤：移除全零行和低频路径
    pivot_flow = pivot_flow.loc[pivot_flow.sum(axis=1) > 0]
    
    print(f"✅ 时间序列矩阵: {pivot_flow.shape[0]} 个时间片, {pivot_flow.shape[1]} 条路径")
    return pivot_flow


# ============================================================
# 3. 相关性计算
# ============================================================
def compute_correlation_matrix(pivot_flow, min_flow=10):
    """
    计算路径间的 Spearman 相关矩阵
    min_flow：路径最小流量阈值（过滤低频路径）
    """
    print(f"📊 过滤低频路径（流量 > {min_flow}）...")
    
    # 仅保留高频路径（流量 > min_flow）
    high_freq_paths = pivot_flow.loc[:, pivot_flow.sum() > min_flow]
    print(f"✅ 保留 {high_freq_paths.shape[1]} 条高频路径")
    
    print("🔗 计算 Spearman 相关矩阵...")
    corr_matrix = high_freq_paths.corr(method='spearman')
    print(f"✅ 相关矩阵计算完成: {corr_matrix.shape}")
    
    return corr_matrix, high_freq_paths


# ============================================================
# 4. 拓扑独立性校验
# ============================================================
def check_topological_independence(path1_sig, path2_sig):
    """
    检查两条路径是否拓扑独立（无共享路段）
    返回 Jaccard 相似度和是否独立的布尔值
    """
    edges1 = set(path1_sig.split('-'))
    edges2 = set(path2_sig.split('-'))
    
    # 计算 Jaccard 相似度
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    jaccard = intersection / union if union > 0 else 0
    
    # 拓扑独立 ↔ 无共享路段 ↔ Jaccard = 0
    is_independent = jaccard == 0
    
    return jaccard, is_independent


# ============================================================
# 5. 地理距离计算
# ============================================================
def get_path_centroid(path_sig, coords_dict):
    """
    计算路径的地理质心（所有路段端点的平均坐标）
    """
    all_coords = []
    for edge_id in path_sig.split('-'):
        if edge_id in coords_dict:
            all_coords.extend(coords_dict[edge_id])
    
    if not all_coords:
        return None
    
    all_coords = np.array(all_coords)
    centroid = np.mean(all_coords, axis=0)
    return centroid


def haversine_distance(coord1, coord2):
    """
    计算两点间的 Haversine 距离（单位：km）
    coord：[lat, lon]
    """
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    R = 6371  # 地球半径（km）
    return R * c


def calculate_path_distance(path1_sig, path2_sig, coords_dict):
    """
    计算两条路径质心间的距离（单位：km）
    """
    centroid1 = get_path_centroid(path1_sig, coords_dict)
    centroid2 = get_path_centroid(path2_sig, coords_dict)
    
    if centroid1 is None or centroid2 is None:
        return None
    
    dist = haversine_distance(centroid1, centroid2)
    return dist


# ============================================================
# 6. 幽灵竞争路径对识别
# ============================================================
def extract_ghost_competition_pairs(corr_matrix, coords_dict, 
                                     min_corr=-0.45, min_dist=1.5, max_jaccard=0.1):
    """
    识别满足以下条件的"幽灵竞争"路径对：
    1. Spearman 相关系数 < min_corr
    2. Jaccard 相似度 <= max_jaccard（拓扑较独立，允许有限共享）
    3. 质心距离 > min_dist km（地理距离远）
    
    参数：
        corr_matrix: Spearman 相关矩阵
        coords_dict: 边坐标映射
        min_corr: 负相关系数阈值，默认 -0.45
        min_dist: 最小地理距离（km），默认 1.5
        max_jaccard: 最大 Jaccard 相似度（0-1），默认 0.1（允许10%的共享路段）
    
    返回 DataFrame，包含所有符合条件的路径对及相关指标
    """
    print(f"\n🔍 寻找幽灵竞争路径对...")
    print(f"   条件："
          f"相关系数 < {min_corr}, "
          f"Jaccard <= {max_jaccard}, "
          f"距离 > {min_dist} km")
    
    results = []
    paths = corr_matrix.columns.tolist()
    
    for i in range(len(paths)):
        if i % max(1, len(paths) // 10) == 0:
            print(f"   进度: {i}/{len(paths)}", end='\r', flush=True)
        
        for j in range(i + 1, len(paths)):
            r = corr_matrix.iloc[i, j]
            
            # 条件 1：强负相关
            if pd.isna(r) or r >= min_corr:
                continue
            
            # 条件 3：地理距离
            dist = calculate_path_distance(paths[i], paths[j], coords_dict)
            if dist is None or dist <= min_dist:
                continue
            
            # 条件 2：拓扑独立性（松弛约束）
            jaccard, is_independent = check_topological_independence(paths[i], paths[j])
            if jaccard > max_jaccard:
                continue
            
            # 所有条件满足：记录这个"幽灵竞争"对
            results.append({
                'path_a': paths[i],
                'path_b': paths[j],
                'corr': r,
                'distance_km': dist,
                'jaccard': jaccard,
            })
    
    print(f"   进度: {len(paths)}/{len(paths)} 完成")
    
    # 处理空结果情形
    if not results:
        print(f"⚠️ 未发现满足条件的幽灵竞争路径对")
        df_ghost = pd.DataFrame(columns=['path_a', 'path_b', 'corr', 'distance_km', 'jaccard'])
    else:
        df_ghost = pd.DataFrame(results).sort_values('corr')
        print(f"✅ 发现 {len(df_ghost)} 个幽灵竞争路径对\n")
    
    return df_ghost


# ============================================================
# 7. 可视化映射
# ============================================================
def visualize_ghost_competition(df_ghost, coords_dict, output_dir):
    """
    在 Folium 地图上可视化"幽灵竞争"路径对
    使用不同颜色虚线表示不同的相关系数强度
    """
    if df_ghost.empty:
        print("⚠️ 无幽灵竞争路径对，跳过可视化")
        return
    
    print("🎨 生成可视化地图...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算所有路径坐标，确定地图中心
    all_lats, all_lons = [], []
    for paths in zip(df_ghost['path_a'], df_ghost['path_b']):
        for path_sig in paths:
            for edge_id in str(path_sig).split('-'):
                if edge_id in coords_dict:
                    for coord in coords_dict[edge_id]:
                        all_lats.append(coord[0])
                        all_lons.append(coord[1])
    
    if not all_lats:
        print("❌ 无有效坐标数据")
        return
    
    center = [np.mean(all_lats), np.mean(all_lons)]
    m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')
    
    # 颜色映射：根据相关系数强度分级
    def get_color_and_weight(corr_val):
        """根据相关系数强度返回颜色和线宽"""
        if corr_val < -0.7:
            return '#8B0000', 3  # 深红
        elif corr_val < -0.6:
            return '#DC143C', 2.5  # 深红
        elif corr_val < -0.5:
            return '#FF6347', 2  # 番茄红
        else:
            return '#FFA500', 1.5  # 橙色
    
    # 绘制每个幽灵竞争对
    for idx, row in df_ghost.iterrows():
        path_a, path_b = row['path_a'], row['path_b']
        corr = row['corr']
        dist = row['distance_km']
        
        color, weight = get_color_and_weight(corr)
        
        # 绘制路径 A
        coords_a = []
        for edge_id in str(path_a).split('-'):
            if edge_id in coords_dict:
                coords_a.extend(coords_dict[edge_id])
        
        if coords_a:
            folium.PolyLine(
                coords_a, color=color, weight=weight, opacity=0.7,
                popup=f"Path A | Corr: {corr:.3f}",
                tooltip=f"Path A (Corr: {corr:.3f})"
            ).add_to(m)
        
        # 绘制路径 B
        coords_b = []
        for edge_id in str(path_b).split('-'):
            if edge_id in coords_dict:
                coords_b.extend(coords_dict[edge_id])
        
        if coords_b:
            folium.PolyLine(
                coords_b, color=color, weight=weight, opacity=0.7,
                dash_array='5, 5',  # 虚线
                popup=f"Path B | Corr: {corr:.3f}",
                tooltip=f"Path B (Corr: {corr:.3f})"
            ).add_to(m)
        
        # 在两条路径质心间连接虚线（表示竞争关系）
        centroid_a = get_path_centroid(path_a, coords_dict)
        centroid_b = get_path_centroid(path_b, coords_dict)
        if centroid_a is not None and centroid_b is not None:
            folium.PolyLine(
                [centroid_a, centroid_b],
                color=color, weight=1, opacity=0.5,
                dash_array='10, 5',
                popup=f"Ghost Competition: r={corr:.3f}, d={dist:.2f}km"
            ).add_to(m)
    
    # 添加图例
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 320px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; opacity: 0.95; border-radius: 5px;">
        <h4 style="margin: 0 0 10px 0;">幽灵竞争分析</h4>
        <p style="margin: 5px 0;"><b>总路径对数:</b> {len(df_ghost)}</p>
        <p style="margin: 5px 0;"><b>相关系数强度</b></p>
        <div style="margin-left: 10px; font-size: 11px;">
            <span style="color:#8B0000;">●</span> r < -0.7 (极强负相关)<br/>
            <span style="color:#DC143C;">●</span> -0.7 < r < -0.6<br/>
            <span style="color:#FF6347;">●</span> -0.6 < r < -0.5<br/>
            <span style="color:#FFA500;">●</span> -0.5 < r (中等负相关)<br/>
        </div>
        <p style="margin: 10px 0 5px 0;"><b>说明</b></p>
        <div style="margin-left: 10px; font-size: 11px;">
            实线：路径主体<br/>
            虚线：竞争关系<br/>
            地理距离 > 1.5km<br/>
            Jaccard 相似度可调<br/>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 保存地图
    output_path = os.path.join(output_dir, "ghost_competition_map.html")
    m.save(output_path)
    print(f"✅ 地图已保存: {output_path}")


# ============================================================
# 8. 主执行逻辑
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="网络级别路径竞争分析 - 幽灵竞争识别")
    parser.add_argument('--min-flow', type=int, default=30, 
                       help='路径最小流量阈值（过滤低频路径）')
    parser.add_argument('--min-corr', type=float, default=-0.45, 
                       help='负相关系数阈值（< 此值才算强负相关）')
    parser.add_argument('--min-dist', type=float, default=1.5, 
                       help='路径质心最小距离阈值（km）')
    parser.add_argument('--max-jaccard', type=float, default=0.1, 
                       help='Jaccard 相似度上限（0-1），允许的最大共享路段比例，默认 0.1')
    parser.add_argument('--input-dir', type=str, default='path_data', 
                       help='输入数据目录')
    parser.add_argument('--graph-file', type=str, default='athens_road_network.graphml', 
                       help='路网 GraphML 文件')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='输出目录')
    args = parser.parse_args()
    
    output_dir = args.output_dir or f"network_analysis/ghost_competition_{datetime.now().strftime('%m%d_%H%M')}"
    
    print("=" * 60)
    print("🌐 网络级别路径竞争分析 - 幽灵竞争识别")
    print("=" * 60)
    
    # 1. 加载数据
    df_all = load_kinematics_data(args.input_dir)
    G, coords_dict = load_graph_and_coords(args.graph_file)
    
    # 2. 数据预处理
    df_all['path_sequence'] = df_all['path_sequence'].apply(parse_path_sequence)
    
    # 3. 构建时间序列矩阵
    pivot_flow = prepare_time_series(df_all)
    
    # 4. 计算相关矩阵
    corr_matrix, high_freq_paths = compute_correlation_matrix(pivot_flow, args.min_flow)
    
    # 5. 识别幽灵竞争对
    df_ghost = extract_ghost_competition_pairs(
        corr_matrix, coords_dict,
        min_corr=args.min_corr,
        min_dist=args.min_dist,
        max_jaccard=args.max_jaccard
    )
    
    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    df_ghost.to_csv(os.path.join(output_dir, "ghost_competition_pairs.csv"), index=False)
    print(f"\n📊 详细数据已保存: {os.path.join(output_dir, 'ghost_competition_pairs.csv')}")
    
    # 7. 可视化
    visualize_ghost_competition(df_ghost, coords_dict, output_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ 分析完成！结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
