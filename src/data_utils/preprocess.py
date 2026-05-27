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
from tqdm import tqdm  
import logging

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

        # 1. 使用 tqdm 包装你的列表
        pbar = tqdm(input_files, desc="路网匹配进度")
        
        # 2. 关键：迭代对象必须是 pbar 而不是 input_files
        for file_path in pbar: 
            file_name = os.path.basename(file_path)
            output_name = file_name.replace('_parsed.parquet', '_matched.parquet')
            output_path = os.path.join(self.processed_dir, output_name)

            if os.path.exists(output_path):
                # 3. 建议：跳过时也打印到 pbar，而不是 logging
                pbar.set_description(f"跳过: {file_name[:10]}")
                continue
            
            # 动态更新进度条右侧的状态文字
            pbar.set_description(f"匹配中: {file_name[:10]}")

            try:
                df = pd.read_parquet(file_path)
                edges = ox.nearest_edges(G, X=df['lon'], Y=df['lat'])
                
                df['u'] = [e[0] for e in edges]
                df['v'] = [e[1] for e in edges]
                df['edge_id'] = df['u'].astype(str) + "_" + df['v'].astype(str)
                
                df.to_parquet(output_path, index=False)

            except Exception as e:
                logging.error(f"❌ 处理文件 {file_name} 时出错: {str(e)}")

        logging.info("所有轨迹已完成路网吸附。")
    
    def step_4_denoise_and_clean(self):
        """
        阶段 4: 轨迹去噪与分段
        """
        from .trajectory import clean_trajectories, segment_trajectories
        
        input_files = glob.glob(os.path.join(self.processed_dir, "*_matched.parquet"))
        clean_dir = os.path.join(self.processed_dir, "cleaned")
        os.makedirs(clean_dir, exist_ok=True)

        for f in tqdm(input_files, desc="去噪与清洗"):
            df = pd.read_parquet(f)
            # 1. 去噪
            df_clean = clean_trajectories(df)
            # 2. 分段
            df_final = segment_trajectories(df_clean)
            
            save_path = os.path.join(clean_dir, os.path.basename(f).replace(".parquet", "_clean.parquet"))
            df_final.to_parquet(save_path, index=False)
        
        self.clean_dir = clean_dir
        logging.info(f"✅ 清洗完成，存储于: {clean_dir}")

    def step_5_statistical_analysis(self):
        """阶段 5: 流量聚合与全自动化质量报告"""
        from .analysis import extract_path_volatility, generate_quality_report, generate_heatmap
        import glob

        # --- 核心修复：确保 clean_dir 属性存在 ---
        if not hasattr(self, 'clean_dir'):
            # 如果没有这个属性，手动构造路径（通常是 data/processed/cleaned）
            self.clean_dir = os.path.join(self.processed_dir, "cleaned")
        # 1. 路径准备
        clean_files = sorted(glob.glob(os.path.join(self.clean_dir, "*.parquet")))
        eda_dir = "eda_results/denoise_verification"
        os.makedirs(eda_dir, exist_ok=True)

        if not clean_files:
            logging.error("❌ 未发现清洗后的数据，请先运行 Step 4")
            return None

        # 2. 循环生成对比图表
        df_list = []
        time_labels = []
        
        for f_clean in clean_files:
            df_c = pd.read_parquet(f_clean)
            df_list.append(df_c)
            
            # 提取标签 (假设文件名格式: pneuma_athens_0830_0900_matched_clean.parquet)
            parts = os.path.basename(f_clean).split('_')
            label = f"{parts[2]}_{parts[3]}"
            time_labels.append(label)

            # 寻找对应的原始匹配文件进行质量对比
            raw_fname = os.path.basename(f_clean).replace("_clean.parquet", ".parquet")
            raw_path = os.path.join(self.processed_dir, raw_fname)
            
            if os.path.exists(raw_path):
                df_r = pd.read_parquet(raw_path)
                report_path = os.path.join(eda_dir, f"{label}_quality_report.png")
                generate_quality_report(df_r, df_c, report_path, title_suffix=label)
                logging.info(f"📊 已生成质量对比报告: {label}")

        # 3. 提取流量矩阵 (用于模型输入)
        flow_matrix, stats = extract_path_volatility(df_list, time_labels)
        
        # 保存结果
        flow_matrix.to_parquet(os.path.join(self.processed_dir, "flow_matrix_T_N.parquet"))
        stats.to_csv(os.path.join(self.processed_dir, "edge_stats.csv"))
        
        # 4. 生成全域流量热力图
        full_coords = pd.concat([df[['lat', 'lon']] for df in df_list], ignore_index=True)
        generate_heatmap(full_coords, os.path.join(eda_dir, "overall_traffic_heatmap.html"))
        
        logging.info(f"✅ 阶段 5 完成！报告位于 {eda_dir}，矩阵位于 data/processed/")
        return flow_matrix
    
    def step_6_extract_path_features(self):
        """
        阶段 6: 路径指纹提取与运动学特征构建
        """
        from .kinematics import extract_path_features
        from .analysis import plot_path_kinematics_report
        import glob
        from tqdm import tqdm

        # 路径初始化
        if not hasattr(self, 'clean_dir'):
            self.clean_dir = os.path.join(self.processed_dir, "cleaned")
        
        data_output_dir = os.path.join(self.processed_dir, "path_features")
        viz_output_dir = "eda_results/path_analysis"
        os.makedirs(data_output_dir, exist_ok=True)
        os.makedirs(viz_output_dir, exist_ok=True)
        
        input_files = sorted(glob.glob(os.path.join(self.clean_dir, "*.parquet")))
        if not input_files:
            logging.error("❌ 阶段 6 失败：未发现清洗后的文件，请先运行 Step 4")
            return

        # 2. 单循环处理：提取 + 保存 + 绘图
        for f_path in tqdm(input_files, desc="阶段 6: 路径特征提取与绘图"):
            # A. 读取数据
            df = pd.read_parquet(f_path)
            
            # B. 执行特征提取
            path_results = extract_path_features(df)
            
            # C. 提取文件名标签 (例如 0830_0900)
            parts = os.path.basename(f_path).split('_')
            base_name = os.path.basename(f_path).replace("_clean.parquet", "")
            fname = f"{base_name}_path_kinematics.parquet"
            
            # D. 保存 Parquet 特征数据
            save_path = os.path.join(data_output_dir, fname)
            path_results.to_parquet(save_path, index=False)
            
            # E. 生成并保存可视化图表 (使用 base_name 作为标签)
            plot_path_kinematics_report(path_results, viz_output_dir, label=base_name)
        
        logging.info(f"✅ 阶段 6 完成！")
        logging.info(f"   - 特征数据: {data_output_dir}")
        logging.info(f"   - 分析图表: {viz_output_dir}")
        
    def step_7_generate_model_ready_data(self):
        """阶段 7: 构建模型输入张量 (支持双通道 Mask 与双图矩阵)"""
        from .generator import PathTensorGenerator # 确保路径正确
        import torch
        import os
        import glob
        from tqdm import tqdm
        import pandas as pd

        # 1. 路径准备
        input_dir = os.path.join(self.processed_dir, "path_features")
        output_path = self.cfg['path']['model_input_pt']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        path_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
        # 在 step_7 中添加打印，确认读取了多少个文件
        path_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
        logging.info(f"📂 阶段 7 正在处理 {len(path_files)} 个特征文件") # 这里应该是 10 个左右
        if not path_files:
            logging.error("❌ 阶段 7 失败：未发现特征文件。")
            return

        # 2. 初始化重构后的生成器
        gen = PathTensorGenerator(self.cfg)
        
        # 3. 构建全局库与邻接矩阵
        global_paths = gen.build_global_path_library(path_files)
        adj_semantic = gen.build_semantic_adj()
        adj_topo = gen.build_topology_adj()
        
        # 4. 生成时空张量 [T, N, 2]
        st_chunks = []
        for f in tqdm(path_files, desc="生成双通道时空张量"):
            df_feat = pd.read_parquet(f)
            # 现在 generate_chunk 返回的是 [T, N, 2]
            chunk = gen.generate_chunk(df_feat)
            st_chunks.append(chunk)

        # 5. 计算流量通道的全局最大值（用于归一化）
        # 注意：只计算通道 0 的最大值
        all_flows = np.concatenate([c[..., 0].flatten() for c in st_chunks])
        max_val = float(np.max(all_flows))
        logging.info(f"📊 流量通道最大值 (Max Val): {max_val}")

        # 6. 封装并保存
        save_data = {
            'x_list': st_chunks,           # 形状列表 [T_i, N, 2]
            'adj_semantic': adj_semantic,   # [N, N]
            'adj_topo': adj_topo,           # [N, N]
            'path_labels': global_paths,
            'max_val': max_val,            # 关键：保存归一化基准
            'config_snapshot': self.cfg['preprocess'] # 保存一份配置快照便于追溯
        }
        
        torch.save(save_data, output_path)
        logging.info(f"✨ 预处理完成！特征数据已存至: {output_path}")
        return output_path