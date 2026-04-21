import os
import pandas as pd
import numpy as np
import folium
import glob
import ast
from scipy.stats import spearmanr
from datetime import datetime

try:
    import networkx as nx
except ImportError:
    raise ImportError("缺少 networkx，请安装：pip install networkx")

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

# ==========================================
# 1. 工具函数
# ==========================================
def load_graph_and_coords(graph_file="athens_road_network.graphml"):
    """
    加载 GraphML 文件并提取边坐标映射
    返回 (G, coords_dict)
    coords_dict: {edge_id: [[lat, lon], [lat, lon], ...], ...}
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
    for u, v, key, data in G.edges(keys=True, data=True) if isinstance(G, nx.MultiDiGraph) else [(u, v, 0, data) for u, v, data in G.edges(data=True)]:
        edge_id = f"{u}_{v}"
        
        # 尝试从 geometry 字段提取坐标
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
        
        # 备用: 直接用节点坐标
        if edge_id not in coords_dict:
            try:
                u_node = G.nodes[u]
                v_node = G.nodes[v]
                u_coords = [float(u_node.get('y', u_node.get('lat', 0))), float(u_node.get('x', u_node.get('lon', 0)))]
                v_coords = [float(v_node.get('y', v_node.get('lat', 0))), float(v_node.get('x', v_node.get('lon', 0)))]
                coords_dict[edge_id] = [u_coords, v_coords]
            except:
                pass
    
    print(f"✅ 提取 {len(coords_dict)} 条边的坐标数据")
    return G, coords_dict

def parse_path_sequence(seq):
    """解析路径序列"""
    if isinstance(seq, (list, tuple, np.ndarray)):
        return list(seq)
    try:
        parsed = ast.literal_eval(str(seq))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except:
        return []

def extract_od_from_path(path_seq):
    """从路径序列提取 OD 对"""
    if not path_seq or len(path_seq) < 2:
        return None, None
    first_edge = str(path_seq[0])
    last_edge = str(path_seq[-1])
    if '_' not in first_edge or '_' not in last_edge:
        return None, None
    origin = first_edge.split('_')[0]
    destination = last_edge.split('_')[-1]
    return origin, destination

# ==========================================
# 2. 核心分析函数
# ==========================================
def generate_competition_map(df_od_all, coords_dict, target_od, output_path, min_corr=-0.3):
    """
    针对单个 OD 对进行负相关分析并绘图
    """
    # 提取并按 15min 切片
    df_od = df_od_all[df_od_all['od_pair'] == target_od].copy()
    df_od['time_bin'] = df_od['start_time'].dt.floor('15T')
    
    if len(df_od) == 0:
        return False, "OD对数据为空"
    
    # 路径流量透视
    pivot_flow = df_od.pivot_table(
        index='time_bin', 
        columns='path_signature', 
        values='track_id', 
        aggfunc='count'
    ).fillna(0)
    
    # 过滤缺失数据（全零行
    pivot_flow = pivot_flow[pivot_flow.sum(axis=1) > 0]
    
    if pivot_flow.shape[0] < 3:  # 数据点太少无法计算相关性
        return False, "有效时间片不足"
    
    if pivot_flow.shape[1] < 2:  # 路径数过少
        return False, "竞争路径不足"

    # 计算 Spearman 相关性
    try:
        corr_matrix = pivot_flow.corr(method='spearman')
    except Exception as e:
        return False, f"相关性计算失败: {e}"
    
    # 寻找负相关最强的路径对
    neg_pairs = []
    paths = corr_matrix.columns
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            r = corr_matrix.iloc[i, j]
            if pd.notna(r) and r <= min_corr:
                neg_pairs.append((paths[i], paths[j], r))
    
    if not neg_pairs:
        return False, "未发现显著负相关路径"
    
    # 取相关性最强的一组
    path_a_sig, path_b_sig, corr_val = sorted(neg_pairs, key=lambda x: x[2])[0]

    # --- 绘图部分 ---
    # 获取起点坐标作为地图中心
    first_edge = str(path_a_sig).split('-')[0]
    center_coords = None
    if first_edge in coords_dict and len(coords_dict[first_edge]) > 0:
        center_coords = coords_dict[first_edge][0]
    else:
        # 使用所有坐标的平均值
        all_lats, all_lons = [], []
        for edge in str(path_a_sig).split('-'):
            if edge in coords_dict:
                for coord in coords_dict[edge]:
                    all_lats.append(coord[0])
                    all_lons.append(coord[1])
        if all_lats:
            center_coords = [np.mean(all_lats), np.mean(all_lons)]
        else:
            return False, "坐标数据缺失"

    m = folium.Map(location=center_coords, zoom_start=13, tiles='CartoDB positron')

    def draw_path(signature, color, name, label):
        all_coords = []
        for edge in str(signature).split('-'):
            if edge in coords_dict:
                all_coords.extend(coords_dict[edge])
        if all_coords:
            folium.PolyLine(
                all_coords, 
                color=color, 
                weight=5, 
                opacity=0.8, 
                tooltip=name,
                popup=folium.Popup(f"<b>{label}</b><br/>Flow Corr: {corr_val:.3f}<br/>Path: {str(signature)[:80]}...", max_width=300)
            ).add_to(m)

    draw_path(path_a_sig, '#0066cc', f'Path A: {path_a_sig[:50]}...', f'Path A (蓝色)')
    draw_path(path_b_sig, '#cc0000', f'Path B: {path_b_sig[:50]}...', f'Path B (红色)')

    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 380px; height: 140px;
                background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
                padding: 12px; opacity: 0.95; border-radius: 5px;">
      <h4 style="margin: 0 0 8px 0;">OD 负相关分析: {target_od}</h4>
      <b>相关系数</b>: {corr_val:.3f}<br/>
      <b>数据点</b>: {pivot_flow.shape[0]} 个时间片<br/>
      <b>竞争路径</b>: {pivot_flow.shape[1]} 条<br/>
      <span style="color:#0066cc;">●</span> <b>Path A</b> (蓝色) vs 
      <span style="color:#cc0000;">●</span> <b>Path B</b> (红色)<br/>
      <i>负相关表明两条路径的选择呈反向关系</i>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    return True, "成功"

# ==========================================
# 3. 执行逻辑
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OD 对负相关路径竞争分析")
    parser.add_argument('--top-n-od', type=int, default=10, help='分析前N个流量最大的OD对')
    parser.add_argument('--min-corr', type=float, default=-0.3, help='负相关系数阈值')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    
    output_dir = args.output_dir or f"path_analysis/negative_corr_{datetime.now().strftime('%m%d_%H%M')}"
    
    # A. 读取数据
    path_files = sorted(glob.glob('path_data/*_path_kinematics.parquet'))
    if not path_files:
        print("❌ 错误：未在 path_data 文件夹下找到 parquet 文件！")
        exit(1)

    print(f"📂 正在读取 {len(path_files)} 个数据文件...")
    df_all = pd.concat([pd.read_parquet(f) for f in path_files], ignore_index=True)
    print(f"✅ 加载 {len(df_all)} 条轨迹记录")
    
    # B. 数据预处理
    df_all['start_time'] = pd.to_datetime(df_all['start_time'])
    df_all['path_sequence'] = df_all['path_sequence'].apply(parse_path_sequence)
    df_all = df_all[df_all['path_sequence'].map(len) > 0].copy()
    
    # 提取 OD 对
    od_info = df_all['path_sequence'].apply(extract_od_from_path)
    df_all[['origin_node', 'destination_node']] = pd.DataFrame(od_info.tolist(), index=df_all.index)
    df_all['od_pair'] = df_all['origin_node'].astype(str) + ' -> ' + df_all['destination_node'].astype(str)
    df_all = df_all.dropna(subset=['od_pair'])
    
    print(f"✅ 提取 {df_all['od_pair'].nunique()} 个唯一OD对")
    
    # C. 加载路网和坐标
    try:
        G, coords_dict = load_graph_and_coords()
    except Exception as e:
        print(f"❌ 路网加载失败: {e}")
        exit(1)
    
    # D. 识别流量最高的OD对
    top_ods = df_all['od_pair'].value_counts().head(args.top_n_od).index.tolist()
    print(f"\n📊 开始分析流量排名前 {args.top_n_od} 的 OD 对，寻找负相关路径对...\n")
    
    success_count = 0
    for idx, od in enumerate(top_ods, 1):
        flow = (df_all['od_pair'] == od).sum()
        print(f"[{idx}/{len(top_ods)}] {od} (流量: {flow}): ", end='', flush=True)
        
        safe_name = od.replace(' -> ', '_').replace(' ', '')
        file_path = os.path.join(output_dir, f"neg_corr_{safe_name}.html")
        
        success, msg = generate_competition_map(
            df_all, coords_dict, od, file_path, 
            min_corr=args.min_corr
        )
        
        if success:
            print(f"✅ {msg} -> {file_path}")
            success_count += 1
        else:
            print(f"⚠️ {msg}")
    
    print(f"\n🎉 任务完成！共生成 {success_count} 个分析图")
    print(f"📁 结果保存在: {output_dir}")