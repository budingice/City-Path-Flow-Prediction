import pandas as pd
import csv
import time
import os
import glob
from datetime import datetime, timedelta

def get_absolute_base_time(file_name):
    """
    从文件名提取绝对时间基准
    示例: 20181024_d1_0830_0900.csv -> 2018-10-24 08:30:00
    """
    try:
        parts = file_name.split('_')
        date_str = parts[0]      # 20181024
        start_time_str = parts[2] # 0830
        base_dt = datetime.strptime(f"{date_str}{start_time_str}", "%Y%m%d%H%M")
        return base_dt
    except Exception as e:
        print(f" 文件名 {file_name} 格式解析失败，将使用默认偏移: {e}")
        return None

def run_batch_parser():
    # --- 配置参数 ---
    input_folder = 'dataset'  # 数据文件夹
    output_folder = 'processed_data' # 处理后保存的文件夹
    sampling_rate = 25 # 25Hz -> 1Hz
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹下所有 csv 文件
    all_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    
    if not all_files:
        print(f"❌ 在 {input_folder} 文件夹下找不到任何 .csv 文件。")
        return

    print(f"🚀 发现 {len(all_files)} 个文件，准备开始批处理...")
    total_start_time = time.time()

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        base_dt = get_absolute_base_time(file_name)
        
        vehicles_list = []
        trajectories_list = []
        
        print(f"\n📄 正在解析: {file_name}")
        file_start_time = time.time()

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            try:
                header = next(reader)
            except StopIteration:
                continue
            
            for row_idx, row in enumerate(reader):
                row = [x.strip() for x in row if x.strip()]
                if len(row) < 10: continue
                
                track_id = int(row[0])
                vehicles_list.append({
                    'track_id': track_id,
                    'type': row[1],
                    'avg_speed': float(row[3])
                })
                
                dynamic_data = row[10:]
                for i in range(0, len(dynamic_data), 6 * sampling_rate):
                    chunk = dynamic_data[i : i + 6]
                    if len(chunk) == 6:
                        rel_time = float(chunk[5])
                        
                        # 核心改进：转换为绝对时间
                        # 如果没有基准时间，则保留相对时间
                        abs_time = base_dt + timedelta(seconds=rel_time) if base_dt else rel_time
                        
                        trajectories_list.append({
                            'track_id': track_id,
                            'lat': float(chunk[0]),
                            'lon': float(chunk[1]),
                            'speed': float(chunk[2]),
                            'timestamp': abs_time # 使用绝对时间戳
                        })

        # 转换为 DataFrame 并保存
        if trajectories_list:
            df_t = pd.DataFrame(trajectories_list)
            output_file = os.path.join(output_folder, file_name.replace('.csv', '.parquet'))
            df_t.to_parquet(output_file, engine='pyarrow')
            
            # 同时保存车辆元数据（可选）
            df_v = pd.DataFrame(vehicles_list)
            df_v.to_parquet(output_file.replace('.parquet', '_info.parquet'), engine='pyarrow')
            
            print(f"✅ {file_name} 解析完成，耗时: {time.time() - file_start_time:.2f}s")
            print(f"📊 轨迹点数: {len(df_t)}")
        else:
            print(f"⚠️ {file_name} 未提取到有效轨迹数据。")

    print(f"\n✨ 所有文件处理完毕！总耗时: {time.time() - total_start_time:.2f} 秒")
    print(f"📂 处理后的数据保存在: {output_folder}")

if __name__ == "__main__":
    run_batch_parser()