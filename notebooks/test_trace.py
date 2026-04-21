import pandas as pd

# 1. 加载 1Hz 的轨迹数据
df = pd.read_pickle('trajectories_1hz.pkl')

# 2. 计算每辆车的统计信息
# 按 track_id 分组，计算时间跨度和数据点数量
stats = df.groupby('track_id').agg(
    start_time=('time', 'min'),
    end_time=('time', 'max'),
    point_count=('track_id', 'count')
)

# 计算轨迹持续时长 (秒)
stats['duration'] = stats['end_time'] - stats['start_time']

# --- A. 找到最晚结束的轨迹 ---
latest_vehicle = stats.sort_values(by='end_time', ascending=False).head(1)
latest_id = latest_vehicle.index[0]
latest_end = latest_vehicle['end_time'].values[0]

# --- B. 找到最长的轨迹 (按持续时间算) ---
longest_duration_vehicle = stats.sort_values(by='duration', ascending=False).head(1)
longest_id = longest_duration_vehicle.index[0]
longest_sec = longest_duration_vehicle['duration'].values[0]

# --- C. 找到点数最多的轨迹 (按空间采样点算) ---
most_points_vehicle = stats.sort_values(by='point_count', ascending=False).head(1)
most_id = most_points_vehicle.index[0]

print("="*50)
print(f"🕒 1. 最晚结束的轨迹 (Latest Finish):")
print(f"   车辆 ID: {latest_id}")
print(f"   结束时间: {latest_end:.2f} 秒 (约 {latest_end/60:.2f} 分钟)")

print(f"\n📏 2. 持续时间最长的轨迹 (Longest Duration):")
print(f"   车辆 ID: {longest_id}")
print(f"   持续时长: {longest_sec:.2f} 秒")
print(f"   时间范围: {longest_duration_vehicle['start_time'].values[0]:.2f} -> {longest_duration_vehicle['end_time'].values[0]:.2f}")

print(f"\n📍 3. 数据点最稠密的轨迹 (Most Points):")
print(f"   车辆 ID: {most_id}")
print(f"   采样点数: {most_points_vehicle['point_count'].values[0]} 个点")
print("="*50)