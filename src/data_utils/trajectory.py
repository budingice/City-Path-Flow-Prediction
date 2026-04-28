import numpy as np
import pandas as pd

def _haversine_vectorized(lat1, lon1, lat2, lon2):
    """批量计算哈弗辛距离 (单位: 米)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c

def clean_trajectories(df: pd.DataFrame, speed_threshold: float = 33.3, drift_threshold: float = 0.5):
    if df.empty: return df.copy()

    df = df.copy()
    # 确保时间格式正确
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(['track_id', 'timestamp']).reset_index(drop=True)

    # 1. 计算前一点的坐标和时间
    # 使用 shift(1) 产生上一行数据，每个 track_id 的第一行为 NaN
    df['lat_prev'] = df.groupby('track_id')['lat'].shift(1)
    df['lon_prev'] = df.groupby('track_id')['lon'].shift(1)
    df['ts_prev'] = df.groupby('track_id')['timestamp'].shift(1)

    # 2. 补全起始点（核心修复点：明确使用 Timedelta）
    df['lat_prev'] = df['lat_prev'].fillna(df['lat'])
    df['lon_prev'] = df['lon_prev'].fillna(df['lon'])
    df['ts_prev'] = df['ts_prev'].fillna(df['timestamp'] - pd.Timedelta(seconds=1))

    # 3. 计算距离 (米)
    dist = _haversine_vectorized(df['lat_prev'].values, df['lon_prev'].values,
                                 df['lat'].values, df['lon'].values)

    # 4. 计算时间差 (秒)
    dt = (df['timestamp'] - df['ts_prev']).dt.total_seconds().replace(0, 1e-6)

    # 5. 计算速度
    inst_speed = dist / dt

    # 6. 过滤逻辑：保留轨迹起点 或 速度合理的点
    mask_new = df['track_id'] != df['track_id'].shift(1)
    valid_mask = (inst_speed <= speed_threshold) & (inst_speed > drift_threshold) | mask_new

    cleaned = df.loc[valid_mask].copy()
    
    # 移除辅助计算列
    cols_to_drop = ['lat_prev', 'lon_prev', 'ts_prev']
    cleaned = cleaned.drop(columns=[c for c in cols_to_drop if c in cleaned.columns])

    return cleaned.reset_index(drop=True)

def segment_trajectories(df: pd.DataFrame, max_gap_seconds: int = 300) -> pd.DataFrame:
    """基于时间间隔对轨迹分段"""
    df = df.copy()
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(['track_id', 'timestamp']).reset_index(drop=True)

    # 1. 计算时间差
    dt_series = df.groupby('track_id')['timestamp'].diff()
    
    # 2. 核心修复：将 Timedelta 转换为总秒数 (float)，再与 int 比较
    # fillna(0) 是为了处理轨迹的第一行（第一行 diff 是 NaN）
    dt_seconds = dt_series.dt.total_seconds().fillna(0)

    # 3. 判断是否为新分段
    # 条件：时间间隔超过阈值 OR 换了车
    new_seg = (dt_seconds > max_gap_seconds) | (df['track_id'] != df['track_id'].shift(1))

    # 4. 生成 ID
    df['segment_id'] = (new_seg.groupby(df['track_id']).cumsum()).astype(int)
    df['segment_key'] = df['track_id'].astype(str) + '_seg' + df['segment_id'].astype(str)
    
    return df