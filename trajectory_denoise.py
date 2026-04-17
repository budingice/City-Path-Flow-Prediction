"""
trajectory_denoise.py

工具：轨迹去噪、分段与热点可视化

函数:
 - clean_trajectories(df, speed_threshold=33.3, drift_threshold=0.5)
 - segment_trajectories(df, max_gap_seconds=300)
 - visualize_hotspots(df, lat_col='lat', lon_col='lon', weight_col='cv', output_html='bottlenecks.html')

假定输入 DataFrame 至少包含列: `track_id`, `timestamp`, `lat`, `lon`。
`timestamp` 会被转换为 pandas datetime 类型。

用法示例:
    python trajectory_denoise.py --input matched_data/snapshot.csv --out cleaned.csv

"""
import os
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False


def _haversine_vectorized(lat1, lon1, lat2, lon2):
    # All inputs in radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c


def clean_trajectories(df: pd.DataFrame, speed_threshold: float = 33.3, drift_threshold: float = 0.5,
                       return_mask: bool = False) -> pd.DataFrame:
    """
    清洗轨迹点：剔除瞬时超速跳跃与静止漂移点。

    参数:
    - df: 包含 `track_id`, `timestamp`, `lat`, `lon` 的 DataFrame
    - speed_threshold: m/s（默认约等于 120 km/h -> 33.3 m/s）
    - drift_threshold: m/s，低于此视为静止/漂移
    - return_mask: 若 True 返回 (df_clean, valid_mask)

    返回:
    - df_clean
    """
    if df.empty:
        if return_mask:
            return df.copy(), pd.Series(dtype=bool)
        return df.copy()

    df = df.copy()
    # ensure timestamp
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(['track_id', 'timestamp']).reset_index(drop=True)

    # groupwise shift
    df['lat_rad'] = np.radians(df['lat'].astype(float))
    df['lon_rad'] = np.radians(df['lon'].astype(float))
    df['lat_prev'] = df.groupby('track_id')['lat_rad'].shift(1)
    df['lon_prev'] = df.groupby('track_id')['lon_rad'].shift(1)
    df['ts_prev'] = df.groupby('track_id')['timestamp'].shift(1)

    # compute distances and dt
    mask_new = df['track_id'] != df['track_id'].shift(1)
    # fill NaN prev values with current to get zero distance
    df['lat_prev'] = df['lat_prev'].fillna(df['lat_rad'])
    df['lon_prev'] = df['lon_prev'].fillna(df['lon_rad'])
    df['ts_prev'] = df['ts_prev'].fillna(df['timestamp'] - pd.Timedelta(seconds=1))

    dist = _haversine_vectorized(df['lat_prev'].values, df['lon_prev'].values,
                                 df['lat_rad'].values, df['lon_rad'].values)
    dt = (df['timestamp'] - df['ts_prev']).dt.total_seconds().replace(0, 1e-6)

    # for new track starts, set dist=0 and dt=1 to avoid huge speeds
    dist[mask_new.values] = 0.0
    dt[mask_new.values] = 1.0

    inst_speed = dist / dt

    # valid if within speed threshold and above drift threshold
    valid_mask = (inst_speed <= speed_threshold) & (inst_speed > drift_threshold)

    # keep the first point of each track (even if speed==0) to preserve track start
    first_points = mask_new
    valid_mask = valid_mask | first_points.values

    cleaned = df.loc[valid_mask].copy()
    # drop helper cols
    cleaned = cleaned.drop(columns=['lat_rad', 'lon_rad', 'lat_prev', 'lon_prev', 'ts_prev'])

    if return_mask:
        return cleaned.reset_index(drop=True), pd.Series(valid_mask, index=df.index)
    return cleaned.reset_index(drop=True)


def segment_trajectories(df: pd.DataFrame, max_gap_seconds: int = 300) -> pd.DataFrame:
    """
    对每个 `track_id` 内基于时间间隔分段，返回带 `segment_id` 的 DataFrame。

    - 当相邻点时间差大于 `max_gap_seconds` 时视为新段。
    """
    df = df.copy()
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(['track_id', 'timestamp']).reset_index(drop=True)

    dt = df.groupby('track_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    new_seg = (dt > max_gap_seconds) | (df['track_id'] != df['track_id'].shift(1))

    # cumulative segment id within each track_id
    df['segment_id'] = (new_seg.groupby(df['track_id']).cumsum()).astype(int)
    # make global segment key
    df['segment_key'] = df['track_id'].astype(str) + '_seg' + df['segment_id'].astype(str)
    return df


def visualize_hotspots(df: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lon',
                       weight_col: str = 'cv', output_html: str = 'bottlenecks.html',
                       center: Tuple[float, float] = (37.977, 23.737), zoom_start: int = 13):
    """
    使用 folium HeatMap 可视化热点。

    - df: 包含经纬度与权重列的 DataFrame
    - weight_col: 若不存在，则热力图使用 1
    """
    if not HAS_FOLIUM:
        raise RuntimeError('folium or folium.plugins.HeatMap not available')

    # prepare heat data
    if weight_col in df.columns:
        heat_data = [[r[lat_col], r[lon_col], float(r[weight_col])] for _, r in df.iterrows()]
    else:
        heat_data = [[r[lat_col], r[lon_col], 1.0] for _, r in df.iterrows()]

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles='cartodbpositron')
    HeatMap(heat_data, radius=10, blur=15, min_opacity=0.2).add_to(m)
    m.save(output_html)
    return output_html


if __name__ == '__main__':
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(description='Trajectory Batch Processor with Analysis')
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--output-dir', default='cleaned_data', help='输出结果目录')
    parser.add_argument('--max-gap', type=int, default=300, help='分段最大间隔(秒)')
    # 重新加回 --make-heat 参数
    parser.add_argument('--make-heat', action='store_true', help='是否生成波动热力图')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取文件列表
    if os.path.isdir(args.input):
        files = sorted(glob(os.path.join(args.input, '*.parquet')) + glob(os.path.join(args.input, '*.parq')))
    else:
        files = [args.input]

    if not files:
        print(f"Error: No files found in {args.input}")
        raise SystemExit(1)

    all_processed_parts = [] # 用于最后生成汇总热力图

    for f in files:
        fname = os.path.basename(f)
        df_in = pd.read_parquet(f)
        
        # 执行清洗
        cleaned = clean_trajectories(df_in)
        segmented = segment_trajectories(cleaned, max_gap_seconds=args.max_gap)
        
        # 保存单个清洗后的文件
        out_path = os.path.join(args.output_dir, fname.replace('.parquet', '_cleaned.parquet'))
        segmented.to_parquet(out_path, index=False)
        print(f"Finished: {fname} -> {len(segmented)} rows")

        if args.make_heat:
            all_processed_parts.append(segmented)

    # 如果需要生成热力图，我们将所有清洗后的数据合并展示热点
    if args.make_heat and all_processed_parts:
        print("Generating unified heatmap...")
        full_df = pd.concat(all_processed_parts, ignore_index=True)
        # 注意：此处热力图权重建议使用流量计数，因为 CV 是路径级的，轨迹点级没有 CV
        out_html = os.path.join(args.output_dir, 'overall_hotspots.html')
        visualize_hotspots(full_df, output_html=out_html)
        print(f"Heatmap saved to {out_html}")