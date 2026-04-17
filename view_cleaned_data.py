"""
view_cleaned_data.py
查看trajectory_denoise.py清洗后的数据

功能:
- 随机查看N辆车的清洗后数据
- 显示数据统计信息
- 可选显示轨迹时间范围、点数等详细信息

用法示例:
    python view_cleaned_data.py --input cleaned_data --num-vehicles 10
    python view_cleaned_data.py --input cleaned_data/file.parquet --num-vehicles 5 --show-stats
"""

import os
import argparse
from glob import glob
import pandas as pd
import numpy as np


def load_cleaned_data(input_path):
    """加载清洗后的数据"""
    if os.path.isfile(input_path) and input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
        return df
    elif os.path.isdir(input_path):
        # 加载目录下所有parquet文件
        files = sorted(glob(os.path.join(input_path, '*_cleaned.parquet')))
        if not files:
            files = sorted(glob(os.path.join(input_path, '*.parquet')))
        
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {input_path}")
        
        print(f"Found {len(files)} parquet file(s), loading all data...")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} rows from {len(files)} file(s)")
        return df
    else:
        raise FileNotFoundError(f"Cannot find {input_path}")


def show_vehicle_details(df, vehicle_id):
    """显示单辆车的详细数据"""
    vehicle_data = df[df['track_id'] == vehicle_id].sort_values('timestamp')
    
    if vehicle_data.empty:
        print(f"  ⚠️ No data found for vehicle {vehicle_id}")
        return None
    
    print(f"\n  🚗 Vehicle ID: {vehicle_id}")
    print(f"  📊 Total points: {len(vehicle_data)}")
    
    if 'segment_id' in vehicle_data.columns:
        num_segments = vehicle_data['segment_id'].max() + 1
        print(f"  📍 Segments: {num_segments}")
    
    time_range = vehicle_data['timestamp'].min(), vehicle_data['timestamp'].max()
    duration = (time_range[1] - time_range[0]).total_seconds()
    print(f"  ⏱️  Time span: {time_range[0]} → {time_range[1]}")
    print(f"  ⏰ Duration: {duration:.0f}s ({duration/60:.1f} min)")
    
    # 计算轨迹统计
    if 'lat' in vehicle_data.columns and 'lon' in vehicle_data.columns:
        lat_range = vehicle_data['lat'].max() - vehicle_data['lat'].min()
        lon_range = vehicle_data['lon'].max() - vehicle_data['lon'].min()
        print(f"  🗺️  Lat range: {lat_range:.6f}° | Lon range: {lon_range:.6f}°")
    
    # 显示样本数据
    print(f"\n  📈 Sample data (first 5 rows):")
    cols_to_show = ['timestamp', 'lat', 'lon']
    if 'segment_id' in vehicle_data.columns:
        cols_to_show.append('segment_id')
    
    sample = vehicle_data[cols_to_show].head()
    for idx, (_, row) in enumerate(sample.iterrows()):
        print(f"    {idx+1}. {row['timestamp']} | Lat:{row['lat']:.6f} Lon:{row['lon']:.6f}", end="")
        if 'segment_id' in cols_to_show:
            print(f" | Seg:{int(row['segment_id'])}", end="")
        print()
    
    return vehicle_data


def show_overall_stats(df):
    """显示整体统计信息"""
    print("\n" + "="*70)
    print("📊 Overall Statistics".center(70))
    print("="*70)
    
    print(f"\n  Total rows: {len(df):,}")
    unique_vehicles = df['track_id'].nunique()
    print(f"  🚗 Unique vehicles: {unique_vehicles:,}")
    
    if 'segment_id' in df.columns:
        unique_segments = df['segment_key'].nunique() if 'segment_key' in df.columns else df.groupby('track_id')['segment_id'].max().sum() + unique_vehicles
        print(f"  📍 Total segments: {unique_segments:,}")
    
    print(f"\n  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # 每辆车的平均点数
    points_per_vehicle = df.groupby('track_id').size()
    print(f"\n  Points per vehicle:")
    print(f"    Mean: {points_per_vehicle.mean():.1f}")
    print(f"    Median: {points_per_vehicle.median():.0f}")
    print(f"    Min: {points_per_vehicle.min()}")
    print(f"    Max: {points_per_vehicle.max()}")
    
    if 'lat' in df.columns and 'lon' in df.columns:
        print(f"\n  Geographic range:")
        print(f"    Lat: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
        print(f"    Lon: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")


def main():
    parser = argparse.ArgumentParser(
        description='View cleaned trajectory data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_cleaned_data.py --input cleaned_data --num-vehicles 10
  python view_cleaned_data.py --input cleaned_data/file.parquet --num-vehicles 5 --show-stats
        """)
    
    parser.add_argument('--input', '-i', required=True, 
                       help='输入目录或文件路径（parquet格式）')
    parser.add_argument('--num-vehicles', '-n', type=int, default=10,
                       help='查看的随机车辆数（默认10）')
    parser.add_argument('--show-stats', action='store_true',
                       help='显示整体统计信息')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可重复性）')
    
    args = parser.parse_args()
    
    # 加载数据
    print("Loading cleaned trajectory data...")
    df = load_cleaned_data(args.input)
    print(f"前10行数据：\n{df.head(10)}")
    # 确保timestamp是datetime类型
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 显示整体统计
    if args.show_stats:
        show_overall_stats(df)
    
    # 随机选择车辆
    print("\n" + "="*70)
    print(f"🎲 Randomly viewing {args.num_vehicles} vehicles".center(70))
    print("="*70)
    
    all_vehicles = df['track_id'].unique()
    
    if args.num_vehicles > len(all_vehicles):
        print(f"⚠️  Requested {args.num_vehicles} vehicles but only {len(all_vehicles)} exist.")
        print(f"   Showing all {len(all_vehicles)} vehicles instead.\n")
        num_to_show = len(all_vehicles)
    else:
        num_to_show = args.num_vehicles
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    selected_vehicles = np.random.choice(all_vehicles, size=num_to_show, replace=False)
    
    for i, vehicle_id in enumerate(selected_vehicles, 1):
        print(f"\n[{i}/{num_to_show}]", end=" ")
        show_vehicle_details(df, vehicle_id)
    
    print("\n" + "="*70)
    print("✅ View completed".center(70))
    print("="*70)


if __name__ == '__main__':
    main()
"""
    python view_cleaned_data.py --input cleaned_data --num-vehicles 10 --show-stats
"""