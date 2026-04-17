"""
    python step4_extract_path.py
    该脚本会从 cleaned_data 目录下的 *_matched.parquet 文件中提取路径运动学序列，并保存到 path_data 目录下。
    输出文件包含路径指纹、平均速度、速度标准差、变异系数
"""

import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm

def extract_path_kinematics():
    input_dir = "cleaned_data" # 清洗后的数据
    output_dir = "path_data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理清洗后的文件
    matched_files = glob.glob(os.path.join(input_dir, "*_cleaned.parquet"))
    
    print(f"🚀 开始提取路径运动学序列，共 {len(matched_files)} 个文件...")

    for file_path in matched_files:
        file_name = os.path.basename(file_path)
        df = pd.read_parquet(file_path)

        # 1. 预处理：确保排序并计算瞬时速度（如果原表没有 speed 列）
        df = df.sort_values(by=['track_id', 'timestamp'])

        # 2. 路径序列提取（去重保留拓扑）
        # 识别路段切换点
        df['edge_changed'] = df['edge_id'] != df.groupby('track_id')['edge_id'].shift()
        
        # 3. 核心修改：按 track_id 聚合路径拓扑 + 运动学指标
        # 我们需要：路径指纹、平均速度、标准差、CV、总时长
        path_results = df.groupby('track_id').agg({
            'edge_id': lambda x: list(dict.fromkeys(x)), # 保持顺序去重，得到路径指纹
            'speed': ['mean', 'std'],                    # 计算路径层面的速度均值和标准差
            'timestamp': ['first', 'last', 'count']      # 记录开始、结束时间及点数（频率）
        })

        # 重命名多级索引列
        path_results.columns = [
            'path_sequence', 'avg_speed', 'std_speed', 
            'start_time', 'end_time', 'point_count'
        ]
        path_results = path_results.reset_index()

        # 4. 计算路径级变异系数 (Path-level CV)
        # CV = 标准差 / 均值
        path_results['path_cv'] = path_results['std_speed'] / path_results['avg_speed']
        
        # 计算路径耗时（秒）
        path_results['duration'] = (path_results['end_time'] - path_results['start_time']).dt.total_seconds()

        # 5. 过滤逻辑增强
        # a. 路径长度至少包含 2 个不同路段
        path_results['path_len'] = path_results['path_sequence'].apply(len)
        # b. 剔除 CV 为 NaN 的数据（通常是只有 1 个点的轨迹）
        path_results = path_results[(path_results['path_len'] >= 2) & (path_results['path_cv'].notna())]

        # 6. 生成路径唯一标识符 (Fingerprint)
        # 方便后续按相同路径进行流量聚合
        path_results['path_signature'] = path_results['path_sequence'].apply(lambda x: "-".join(map(str, x)))

        # 7. 保存结果
        output_file = os.path.join(output_dir, file_name.replace("_matched", "_path_kinematics"))
        path_results.to_parquet(output_file)
        print(f"✅ 已保存: {os.path.basename(output_file)} (包含 {len(path_results)} 条带 CV 的路径)")

if __name__ == "__main__":
    extract_path_kinematics()
    