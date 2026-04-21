import os
import csv
import glob
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np

class TrafficDataPipeline:
    def __init__(self, config):
        self.cfg = config
        # 从配置中获取路径
        self.raw_data_dir = config['path']['raw_data_dir'] # 原 'dataset' 文件夹
        self.processed_dir = config['path']['processed_dir']
        self.sampling_rate = config['preprocess'].get('sampling_rate', 25)

    def _get_absolute_base_time(self, file_name):
        """内部工具函数：从文件名提取绝对时间基准"""
        try:
            parts = file_name.split('_')
            date_str = parts[0]
            start_time_str = parts[2]
            return datetime.strptime(f"{date_str}{start_time_str}", "%Y%m%d%H%M")
        except Exception as e:
            logging.warning(f"文件名 {file_name} 格式解析失败: {e}")
            return None

    def step_1_parse_pneuma(self):
        """
        对应你原有的 Step 1: 原始 CSV 转 Parquet
        """
        all_files = sorted(glob.glob(os.path.join(self.raw_data_dir, "*.csv")))
        if not all_files:
            logging.error(f"在 {self.raw_data_dir} 下找不到 .csv 文件")
            return

        logging.info(f"🚀 开始解析 pNEUMA 原始数据，共 {len(all_files)} 个文件...")
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            base_dt = self._get_absolute_base_time(file_name)
            trajectories_list = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader, None)
                if header is None: continue
                
                for row in reader:
                    row = [x.strip() for x in row if x.strip()]
                    if len(row) < 10: continue
                    
                    track_id = int(row[0])
                    dynamic_data = row[10:]
                    
                    # 按照采样率步长提取数据 (25Hz -> 1Hz)
                    for i in range(0, len(dynamic_data), 6 * self.sampling_rate):
                        chunk = dynamic_data[i : i + 6]
                        if len(chunk) == 6:
                            rel_time = float(chunk[5])
                            abs_time = base_dt + timedelta(seconds=rel_time) if base_dt else rel_time
                            
                            trajectories_list.append({
                                'track_id': track_id,
                                'lat': float(chunk[0]),
                                'lon': float(chunk[1]),
                                'speed': float(chunk[2]),
                                'timestamp': abs_time
                            })

            if trajectories_list:
                df = pd.DataFrame(trajectories_list)
                # 保存到新仓库的 data/processed 目录下
                output_name = file_name.replace('.csv', '_parsed.parquet')
                save_path = os.path.join(self.processed_dir, output_name)
                df.to_parquet(save_path, engine='pyarrow')
                logging.info(f"✅ 已生成中间件: {output_name}")

    def visualize_sampling_tracks(self, num_tracks=10):
        """
        对应原 Step 2: 随机抽取车辆轨迹并在地图上叠加显示
        """
        logging.info(f"正在准备地图可视化（采样车辆数: {num_tracks}）...")
        
        # 1. 查找已处理的 parquet 文件
        parquet_files = glob.glob(os.path.join(self.processed_dir, "*_parsed.parquet"))
        if not parquet_files:
            logging.error("找不到已解析的轨迹文件，请先运行 step_1_parse_pneuma")
            return
        
        # 2. 读取路网 (路径由 config 提供)
        graph_file = self.cfg['path'].get('graph_file', 'data/raw/athens_road_network.graphml')
        if os.path.exists(graph_file):
            G = ox.load_graphml(graph_file)
        else:
            logging.warning("本地路网文件不存在，尝试在线下载...")
            # 这里的逻辑可以根据 config 里的中心点下载
            G = ox.graph_from_place("Athens, Greece", network_type='drive')

        # 3. 加载第一个解析后的文件进行预览
        df_t = pd.read_parquet(parquet_files[0])
        sample_tracks = df_t['track_id'].unique()[:num_tracks]
        df_sample = df_t[df_t['track_id'].isin(sample_tracks)]

        # 4. 绘图逻辑
        fig, ax = ox.plot_graph(G, show=False, close=False, edge_color='#555555', 
                                edge_linewidth=0.8, node_size=0, bgcolor='white')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_tracks)))
        for tid, color in zip(sample_tracks, colors):
            track_data = df_sample[df_sample['track_id'] == tid].sort_values('timestamp')
            ax.scatter(track_data['lon'], track_data['lat'], s=5, color=color, zorder=3, alpha=0.7)

        plt.title(f"Athens Traffic Visualization (Sample: {num_tracks} vehicles)")
        
        # 自动保存到结果目录
        save_path = os.path.join(self.processed_dir, "visualization_preview.png")
        plt.savefig(save_path)
        logging.info(f"可视化预览图已保存至: {save_path}")
        plt.show()
    
    def step_3_map_matching(self):
        """
        对应原 Step 3: 将经纬度轨迹匹配到路网边缘
        """
        # 1. 检查并加载路网
        graph_file = self.cfg['path'].get('graph_file')
        if not os.path.exists(graph_file):
            logging.error(f"找不到路网文件: {graph_file}")
            return

        logging.info("📍 正在加载路网模型进行地图匹配...")
        G = ox.load_graphml(graph_file)

        # 2. 获取待处理文件
        # 注意：这里读的是 step_1 产出的 _parsed.parquet
        input_files = glob.glob(os.path.join(self.processed_dir, "*_parsed.parquet"))
        
        if not input_files:
            logging.error("没有找到已解析的轨迹文件，请确认 Step 1 已成功运行")
            return

        # 3. 执行匹配循环
        for file_path in input_files:
            file_name = os.path.basename(file_path)
            # 定义匹配后的文件名，防止重复处理
            output_name = file_name.replace('_parsed.parquet', '_matched.parquet')
            output_path = os.path.join(self.processed_dir, output_name)

            if os.path.exists(output_path):
                logging.info(f"⏭️  文件已匹配，跳过: {output_name}")
                continue

            logging.info(f"--- 正在匹配轨迹: {file_name} ---")
            df = pd.read_parquet(file_path)

            try:
                # 向量化计算最近的路段 (u, v, key)
                # X 为经度 lon, Y 为纬度 lat
                edges = ox.nearest_edges(G, X=df['lon'], Y=df['lat'])
                
                # 提取起点 u 和终点 v
                df['u'] = [e[0] for e in edges]
                df['v'] = [e[1] for e in edges]
                
                # 生成唯一的边 ID (用于后续 Path Flow 统计)
                df['edge_id'] = df['u'].astype(str) + "_" + df['v'].astype(str)
                
                # 保存匹配结果
                df.to_parquet(output_path, index=False)
                logging.info(f"✅ 匹配完成: {output_name}")

            except Exception as e:
                logging.error(f"❌ 处理文件 {file_name} 时出错: {str(e)}")

        logging.info("所有轨迹已完成路网吸附。")
        # 后续你会把 Step 2-3, 4-5 的逻辑也写进这个类中...