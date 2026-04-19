"""analyze_path_competition_map.py

生成 OD 对热力图地图，用于可视化路径竞争和热门 OD 对流量分布。

功能:
- 读取 path_data 中的 *_path_kinematics.parquet 数据
- 提取 OD 对及不同路径指纹
- 统计流量最高且路径最多的 OD 对
- 加载雅典路网 GraphML
- 为每个 Top OD 对生成交互式 Folium HTML 地图

用法示例:
    python analyze_path_competition_map.py --top-n-od 5 --top-map-ods 3 --min-flow 5
"""

import os
import glob
import argparse
import ast
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("缺少 networkx，请安装：pip install networkx") from exc

try:
    import folium
    from folium.plugins import HeatMap
except ImportError as exc:
    raise ImportError("缺少 folium，请安装：pip install folium") from exc

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


def load_kinematics_data(input_dir="path_data"):
    files = sorted(glob.glob(os.path.join(input_dir, "*_path_kinematics.parquet")))
    if not files:
        raise FileNotFoundError(f"未发现路径数据文件: {input_dir}")

    print(f"📂 加载 {len(files)} 个轨迹文件...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    return df


def parse_path_sequence(path_sequence):
    if isinstance(path_sequence, (list, tuple, np.ndarray)):
        return list(path_sequence)
    if pd.isna(path_sequence):
        return []
    try:
        parsed = ast.literal_eval(str(path_sequence))
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)
    except Exception:
        pass
    if isinstance(path_sequence, str):
        return [path_sequence]
    return []


def extract_od_pair(path_sequence):
    if not path_sequence:
        return None, None, None

    first_edge = str(path_sequence[0])
    last_edge = str(path_sequence[-1])
    if '_' in first_edge and '_' in last_edge:
        origin = first_edge.split('_')[0]
        destination = last_edge.split('_')[-1]
        return origin, destination, f"{origin} -> {destination}"

    return None, None, None


def prepare_od_data(df):
    df['path_sequence'] = df['path_sequence'].apply(parse_path_sequence)
    df = df[df['path_sequence'].map(len) > 0].copy()

    od_info = df['path_sequence'].apply(extract_od_pair)
    df[['origin_node', 'destination_node', 'od_pair']] = pd.DataFrame(od_info.tolist(), index=df.index)
    df = df.dropna(subset=['od_pair']).copy()
    return df


def get_top_od_pairs(df, top_n=5, min_flow=3):
    od_stats = (
        df.groupby('od_pair')
          .agg(total_flow=('track_id', 'count'), num_paths=('path_signature', 'nunique'))
          .reset_index()
    )
    filtered = od_stats[od_stats['total_flow'] >= min_flow]
    selected = filtered.sort_values(['total_flow', 'num_paths'], ascending=[False, False]).head(top_n)
    return selected


def load_graph(graph_file="athens_road_network.graphml"):
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"未找到路网文件: {graph_file}")
    print(f"🗺️ 读取路网: {graph_file}")
    if HAS_OSMNX:
        return ox.load_graphml(graph_file)
    return nx.read_graphml(graph_file)


def edge_to_coordinates(G, edge_id):
    edge_id = str(edge_id).strip()
    if '_' not in edge_id:
        return []

    u_str, v_str = edge_id.split('_', 1)
    try:
        u = int(u_str)
        v = int(v_str)
    except ValueError:
        return []

    if not G.has_node(u) or not G.has_node(v):
        return []

    # 支持 MultiDiGraph 或 DiGraph
    edge_data = None
    if G.has_edge(u, v):
        edge_data = G.get_edge_data(u, v)
    elif G.has_edge(v, u):
        edge_data = G.get_edge_data(v, u)
    else:
        return []

    if edge_data is None:
        return []

    if isinstance(edge_data, dict) and 0 in edge_data:
        edge_data = edge_data[0]
    if not isinstance(edge_data, dict):
        return []

    geom = edge_data.get('geometry')
    if geom is not None:
        if hasattr(geom, 'coords'):
            return [[lat, lon] for lon, lat in geom.coords]
        if isinstance(geom, str) and geom.startswith('LINESTRING'):
            coords = []
            items = geom.replace('LINESTRING(', '').replace(')', '').split(',')
            for item in items:
                lon, lat = map(float, item.strip().split())
                coords.append([lat, lon])
            return coords

    u_node = G.nodes[u]
    v_node = G.nodes[v]
    try:
        return [[float(u_node['y']), float(u_node['x'])], [float(v_node['y']), float(v_node['x'])]]
    except Exception:
        try:
            return [[float(u_node['lat']), float(u_node['lon'])], [float(v_node['lat']), float(v_node['lon'])]]
        except Exception:
            return []


def path_signature_to_coords(G, path_signature):
    edge_ids = str(path_signature).split('-')
    full_coords = []
    for edge_id in edge_ids:
        coords = edge_to_coordinates(G, edge_id)
        if not coords:
            continue
        if full_coords and coords[0] == full_coords[-1]:
            full_coords.extend(coords[1:])
        else:
            full_coords.extend(coords)
    return full_coords


def path_signature_to_edge_segments(G, path_signature):
    edge_ids = str(path_signature).split('-')
    segments = []
    for edge_id in edge_ids:
        coords = edge_to_coordinates(G, edge_id)
        if coords:
            segments.append(coords)
    return segments


def build_heatmap_for_od(df, G, od_pair, max_paths=5):
    od_df = df[df['od_pair'] == od_pair]
    if od_df.empty:
        return None

    path_flow = (
        od_df['path_signature']
             .value_counts()
             .rename_axis('path_signature')
             .reset_index(name='flow')
    )
    if path_flow.empty:
        return None

    max_flow = path_flow['flow'].max()
    heat_data = []
    line_paths = []

    for rank, row in path_flow.iterrows():
        coords = path_signature_to_coords(G, row['path_signature'])
        if len(coords) < 2:
            continue
        intensity = float(row['flow']) / float(max_flow)
        for lat, lon in coords:
            heat_data.append([lat, lon, max(intensity, 0.05)])
        if len(line_paths) < max_paths:
            line_paths.append((row['path_signature'], row['flow'], coords))

    if not heat_data:
        return None
    return {'heat_data': heat_data, 'line_paths': line_paths, 'total_flow': int(path_flow['flow'].sum()), 'path_count': int(len(path_flow))}


def create_html_heatmap(od_pair, od_meta, output_dir, G):
    heat_data = od_meta['heat_data']
    line_paths = od_meta['line_paths']
    total_flow = od_meta['total_flow']
    path_count = od_meta['path_count']

    latitudes = [p[0] for p in heat_data]
    longitudes = [p[1] for p in heat_data]
    center = [float(np.mean(latitudes)), float(np.mean(longitudes))]

    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')
    HeatMap(heat_data, radius=14, blur=12, max_zoom=13, min_opacity=0.3).add_to(m)

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    origin_color = '#000000'
    destination_color = '#8c564b'
    mid_opacity = 0.8

    for idx, (path_signature, flow, coords) in enumerate(line_paths):
        base_color = base_colors[idx % len(base_colors)]
        segments = path_signature_to_edge_segments(G, path_signature)
        if not segments:
            # fallback to drawing the whole path
            folium.PolyLine(
                locations=coords,
                color=base_color,
                weight=4,
                opacity=0.8,
                tooltip=f"Path {idx+1} | Flow {flow}",
                popup=folium.Popup(
                    f"<b>OD:</b> {od_pair}<br/>"
                    f"<b>Path rank:</b> {idx+1}<br/>"
                    f"<b>Flow:</b> {flow}<br/>"
                    f"<b>Path signature:</b><br/>{path_signature[:120]}...",
                    max_width=400,
                ),
            ).add_to(m)
            continue

        for seg_idx, seg_coords in enumerate(segments):
            seg_color = base_color
            if seg_idx == 0:
                seg_color = origin_color
            elif seg_idx == len(segments) - 1:
                seg_color = destination_color

            folium.PolyLine(
                locations=seg_coords,
                color=seg_color,
                weight=5 if seg_idx in (0, len(segments) - 1) else 4,
                opacity=mid_opacity,
                tooltip=f"OD: {od_pair} | Path {idx+1} | Segment {seg_idx+1} | Flow {flow}",
            ).add_to(m)

        # add a marker for the origin and destination of this path
        if coords:
            folium.CircleMarker(
                location=coords[0],
                radius=5,
                color=origin_color,
                fill=True,
                fill_color=origin_color,
                fill_opacity=1.0,
                popup=f"起点 Path {idx+1}",
            ).add_to(m)
            folium.CircleMarker(
                location=coords[-1],
                radius=5,
                color=destination_color,
                fill=True,
                fill_color=destination_color,
                fill_opacity=1.0,
                popup=f"终点 Path {idx+1}",
            ).add_to(m)

    path_legend_lines = ''.join([
        f"<span style='color:{base_colors[i]};'>●</span> Path {i+1}<br/>"
        for i in range(min(len(line_paths), len(base_colors)))
    ])
    legend_html = f'''
    <div style="position: fixed; bottom: 45px; left: 45px; width: 340px; height: 220px;
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px; opacity: 0.95; overflow: auto;">
      <b>OD 对</b>: {od_pair}<br/>
      <b>总样本量</b>: {total_flow} 车次<br/>
      <b>独立路径数</b>: {path_count}<br/>
      <span style="color:{origin_color};">●</span> 起始路径段 (Origin)<br/>
      <span style="color:{destination_color};">●</span> 结束路径段 (Destination)<br/>
      <span style="color:{base_colors[0]};">●</span> 中间路径段 (Intermediate)<br/>
      <br/>
      <b>路径颜色</b>:<br/>
      {path_legend_lines}
      <br/>
      <i>不同路径使用不同颜色，首末段用黑色/棕色区分。</i>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    safe_name = od_pair.replace(' -> ', '_').replace(' ', '').replace(':', '')
    output_path = os.path.join(output_dir, f"od_heatmap_{safe_name}.html")
    m.save(output_path)
    return output_path


def generate_od_heatmaps(df, graph_file, output_dir, top_n_od=5, top_map_ods=3, min_flow=3):
    df = prepare_od_data(df)
    od_top = get_top_od_pairs(df, top_n=top_n_od, min_flow=min_flow)
    if od_top.empty:
        raise ValueError("未找到满足条件的 OD 对，请降低 min_flow 或检查数据。")

    os.makedirs(output_dir, exist_ok=True)
    od_top.to_csv(os.path.join(output_dir, "top_od_pairs.csv"), index=False)

    print(f"📊 选取 Top {top_map_ods} 个 OD 对进行地图热力图渲染")
    selected = od_top.head(top_map_ods)['od_pair'].tolist()

    G = load_graph(graph_file)

    output_files = []
    for od_pair in selected:
        od_meta = build_heatmap_for_od(df, G, od_pair)
        if od_meta is None:
            print(f"⚠️ 未能为 OD 对 {od_pair} 生成热力图：无有效坐标")
            continue
        path_html = create_html_heatmap(od_pair, od_meta, output_dir, G)
        print(f"✅ 已保存 {od_pair} 的 HTML 地图: {path_html}")
        output_files.append(path_html)

    return output_files


def main():
    parser = argparse.ArgumentParser(description="生成 OD 对路径竞争的 Folium HTML 热力图")
    parser.add_argument('--top-n-od', type=int, default=10, help='统计时考虑的 Top N 个 OD 对')
    parser.add_argument('--top-map-ods', type=int, default=3, help='生成 HTML 地图的 OD 对数量')
    parser.add_argument('--min-flow', type=int, default=5, help='过滤低流量 OD 对的最小样本数')
    parser.add_argument('--input-dir', type=str, default='path_data', help='path_data 数据目录')
    parser.add_argument('--graph-file', type=str, default='athens_road_network.graphml', help='雅典路网 GraphML 文件')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    output_dir = args.output_dir or f"analysis_results/od_heatmap_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(output_dir, exist_ok=True)

    df = load_kinematics_data(args.input_dir)
    html_paths = generate_od_heatmaps(
        df=df,
        graph_file=args.graph_file,
        output_dir=output_dir,
        top_n_od=args.top_n_od,
        top_map_ods=args.top_map_ods,
        min_flow=args.min_flow,
    )

    print(f"\n🎉 生成完成，共输出 {len(html_paths)} 个 HTML 地图")
    print(f"📁 查看目录: {output_dir}")
    for html in html_paths:
        print(f"  - {html}")


if __name__ == '__main__':
    main()
