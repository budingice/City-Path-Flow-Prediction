import pandas as pd
import numpy as np

def extract_path_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    核心逻辑：从清洗后的轨迹点提取路径指纹及运动学特征
    """
    # 1. 确保排序
    df = df.sort_values(by=['track_id', 'timestamp'])

    # 2. 识别速度列 (适配不同版本的命名)
    s_col = 'speed' if 'speed' in df.columns else 'inst_speed'

    # 3. 按 track_id 聚合路径拓扑 + 运动学指标
    path_results = df.groupby('track_id').agg({
        'edge_id': lambda x: list(dict.fromkeys(x)),      # 路径指纹 (保持顺序去重)
        s_col: ['mean', 'std'],                           # 速度均值与标准差
        'timestamp': ['first', 'last', 'count']           # 时间窗口与点数
    })

    # 重命名多级索引列
    path_results.columns = [
        'path_sequence', 'avg_speed', 'std_speed', 
        'start_time', 'end_time', 'point_count'
    ]
    path_results = path_results.reset_index()

    # 4. 计算路径级特征
    # 计算变异系数 (CV)
    path_results['path_cv'] = path_results['std_speed'] / (path_results['avg_speed'] + 1e-6)
    # 计算耗时 (秒)
    path_results['duration'] = (path_results['end_time'] - path_results['start_time']).dt.total_seconds()
    # 计算路径经过的路段数
    path_results['path_len'] = path_results['path_sequence'].apply(len)

    # 5. 生成路径唯一标识符 (Signature)
    path_results['path_signature'] = path_results['path_sequence'].apply(lambda x: "-".join(map(str, x)))

    # 6. 过滤逻辑：至少包含 2 个路段且 CV 有效
    path_results = path_results[(path_results['path_len'] >= 2) & (path_results['path_cv'].notna())]

    return path_results