"""
plot_athens_top_paths.py
生成雅典地区 Top 路径展示地图

功能:
- 从 path_data 读取 path_kinematics parquet 文件
- 统计热门路径并选取 Top N 路径
- 使用 OSMnx 加载 Athens 驾驶路网
- 将路径 edge_id 转换为地图线条并绘制到 Folium HTML 地图

用法示例:
    python plot_athens_top_paths.py
"""

import os
import glob
import pandas as pd
import numpy as np

try:
    import osmnx as ox
    import folium
    from folium import plugins
except ImportError as exc:
    raise ImportError(
        "缺少依赖，请先安装 osmnx 和 folium: pip install osmnx folium"
    ) from exc


class PathMapGenerator:
    def __init__(self, input_dir="path_data", place_name="Athens, Greece", graph_file="athens_road_network.graphml"):
        self.input_dir = input_dir
        self.place_name = place_name
        self.graph_file = graph_file
        self.all_data = self._load_kinematics_data()
        self.G = None
        self.gdf_nodes = None
        self.gdf_edges = None

    def _load_kinematics_data(self):
        """加载所有路径运动学数据"""
        files = sorted(glob.glob(os.path.join(self.input_dir, "*_path_kinematics.parquet")))
        if not files:
            raise FileNotFoundError(
                f"未发现 path_data 目录下的 *_path_kinematics.parquet 文件: {self.input_dir}"
            )

        print(f"📂 正在读取 {len(files)} 个路径运动学文件...")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ 已加载 {len(df)} 条路径记录")
        return df

    def _athens_bbox(self):
        """返回雅典区域的近似边界框: (north, south, east, west)"""
        return 38.10, 37.86, 23.90, 23.60

    def _load_graph(self):
        """加载 OSM 驾驶路网"""
        if self.G is None:
            # 优先加载本地 GraphML 文件
            if os.path.exists(self.graph_file):
                print(f"🏠 正在加载本地路网: {self.graph_file}...")
                self.G = ox.load_graphml(self.graph_file)
                print(f"✅ 本地路网加载完成。节点: {len(self.G.nodes)}, 边: {len(self.G.edges)}")
                return self.G
            
            print(f"🌍 正在加载 {self.place_name} 的驾驶路网...")
            try:
                self.G = ox.graph_from_place(self.place_name, network_type='drive')
                print("✅ 使用 graph_from_place 成功加载路网")
            except Exception as exc:
                print(f"⚠️ graph_from_place 加载失败: {exc}")
                print("🔄 尝试使用雅典边界框加载路网...")
                north, south, east, west = self._athens_bbox()
                self.G = ox.graph_from_bbox((north, south, east, west), network_type='drive')
                print("✅ 使用 graph_from_bbox 成功加载路网")
        
        return self.G

    def _get_gdfs(self):
        """获取路网的节点和边 GeoDataFrame"""
        if self.gdf_nodes is None or self.gdf_edges is None:
            G = self._load_graph()
            self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(G)
        return self.gdf_nodes, self.gdf_edges

    def _find_edge_coordinates(self, edge_id):
        """
        查找边在路网中的坐标。
        edge_id 格式: 'u_v'（如 '250691723_250691755'）
        """
        try:
            G = self._load_graph()
            
            # 解析 edge_id
            parts = str(edge_id).split('_')
            if len(parts) != 2:
                return []
            
            u_str, v_str = parts
            try:
                u, v = int(u_str), int(v_str)
            except ValueError:
                return []
            
            # 尝试直接查询边
            if G.has_edge(u, v):
                # 处理 MultiDiGraph（可能有多条边）
                edge_data = None
                try:
                    edge_keys = list(G[u][v].keys())
                    if edge_keys:
                        edge_data = G[u][v][edge_keys[0]]
                except:
                    edge_data = G[u][v]
                
                if edge_data:
                    geom = edge_data.get('geometry')
                    if geom is not None:
                        coords = [[lat, lon] for lon, lat in geom.coords]
                        return coords
                    else:
                        u_node = G.nodes[u]
                        v_node = G.nodes[v]
                        return [[u_node['y'], u_node['x']], [v_node['y'], v_node['x']]]
            
            # 反向边
            elif G.has_edge(v, u):
                edge_data = None
                try:
                    edge_keys = list(G[v][u].keys())
                    if edge_keys:
                        edge_data = G[v][u][edge_keys[0]]
                except:
                    edge_data = G[v][u]
                
                if edge_data:
                    geom = edge_data.get('geometry')
                    if geom is not None:
                        coords = [[lat, lon] for lon, lat in geom.coords]
                        return coords[::-1]
                    else:
                        u_node = G.nodes[u]
                        v_node = G.nodes[v]
                        return [[v_node['y'], v_node['x']], [u_node['y'], u_node['x']]]
        except Exception as e:
            pass
        
        return []

    def get_top_path_signatures(self, top_n=50):
        """获取 Top N 路径指纹"""
        counts = self.all_data['path_signature'].value_counts()
        return counts.index.tolist()[:top_n]

    def generate_html_map(self, output_dir="maps", top_n_to_plot=30, save_html=True):
        """生成交互式 HTML 地图"""
        top_paths = self.get_top_path_signatures(top_n=top_n_to_plot)
        if not top_paths:
            raise ValueError("未找到任何 Top Path Signature。请检查 path_data 数据。")

        print(f"\n获取地图中心坐标...")
        gdf_nodes, gdf_edges = self._get_gdfs()
        map_center = [gdf_nodes['y'].mean(), gdf_nodes['x'].mean()]
        print(f"📍 地图中心: {map_center}")

        # 创建地图
        m = folium.Map(
            location=map_center,
            zoom_start=13,
            tiles='CartoDB positron'
        )
        minimap = plugins.MiniMap()
        m.add_child(minimap)

        # 颜色列表
        colors = ['#3186cc', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

        print(f"🖌️ 正在绘制 Top {len(top_paths)} 条路径至地图...")

        success_count = 0
        for i, path_sig in enumerate(top_paths):
            edge_ids = path_sig.split('-')
            full_coords = []
            
            for edge_id in edge_ids:
                coords = self._find_edge_coordinates(edge_id)
                if not coords:
                    continue
                
                # 合并相邻边的坐标
                if full_coords and len(coords) > 0:
                    # 如果当前边的起点与上一条边的终点相同，则只添加除起点外的部分
                    if full_coords[-1] == coords[0]:
                        full_coords.extend(coords[1:])
                    else:
                        full_coords.extend(coords)
                else:
                    full_coords.extend(coords)

            if not full_coords or len(full_coords) < 2:
                print(f"  ⚠️ 路径 {i+1} 未找到足够的边坐标，跳过。")
                continue

            path_color = colors[i % len(colors)]
            flow_count = len(self.all_data[self.all_data['path_signature'] == path_sig])
            
            folium.PolyLine(
                locations=full_coords,
                color=path_color,
                weight=3,
                opacity=0.7,
                tooltip=f"Path Rank: {i+1} | Edges: {len(edge_ids)} | Flow: {flow_count}",
                popup=folium.Popup(
                    f"<b>路径排名:</b> {i+1}<br>"
                    f"<b>流量数:</b> {flow_count}<br>"
                    f"<b>路段数:</b> {len(edge_ids)}<br>"
                    f"<b>路径指纹:</b><br>{path_sig[:100]}...",
                    max_width=400
                )
            ).add_to(m)
            success_count += 1
            print(f"  ✅ 已绘制路径 {i+1}: {len(full_coords)} 个坐标点, 流量 {flow_count}")

        print(f"\n✨ 成功绘制 {success_count} 条路径")

        if save_html:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "athens_top_paths_map.html")
            m.save(output_path)
            print(f"💾 HTML 地图已保存至: {output_path}")
            print(f"💡 请用浏览器打开该文件进行交互式查看")
        
        return m


if __name__ == '__main__':
    try:
        print("=" * 70)
        print("🗺️ 雅典 Top 路径地图生成器")
        print("=" * 70)
        
        generator = PathMapGenerator(
            input_dir='path_data',
            place_name='Athens, Greece',
            graph_file='athens_road_network.graphml'
        )
        generator.generate_html_map(output_dir='maps', top_n_to_plot=30, save_html=True)
        
        print("\n" + "=" * 70)
        print("✅ 地图生成完成！")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
