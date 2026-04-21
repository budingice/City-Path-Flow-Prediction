import pandas as pd
import numpy as np
import csv
import time
import os
import glob
from datetime import datetime, timedelta

# ==========================================
# 模块 1: 核心解析逻辑
# ==========================================
def get_absolute_base_time(file_name):
    """从文件名解析绝对时间基准"""
    try:
        parts = file_name.split('_')
        return datetime.strptime(f"{parts[0]}{parts[2]}", "%Y%m%d%H%M")
    except:
        return None

def run_batch_parser(input_folder='dataset', output_folder='processed_data', sampling_rate=25):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    print(f"🚀 发现 {len(all_files)} 个原始文件，开始解析...")

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        base_dt = get_absolute_base_time(file_name)
        trajectories_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader) # skip header
            for row in reader:
                row = [x.strip() for x in row if x.strip()]
                if len(row) < 10: continue
                
                track_id = int(row[0])
                dynamic_data = row[10:]
                for i in range(0, len(dynamic_data), 6 * sampling_rate):
                    chunk = dynamic_data[i : i + 6]
                    if len(chunk) == 6:
                        rel_time = float(chunk[5])
                        abs_time = base_dt + timedelta(seconds=rel_time) if base_dt else rel_time
                        trajectories_list.append({
                            'track_id': track_id,
                            'lat': float(chunk[0]),
                            'lon': float(chunk[1]),
                            'timestamp': abs_time
                        })
        
        df_t = pd.DataFrame(trajectories_list)
        out_p = os.path.join(output_folder, file_name.replace('.csv', '.parquet'))
        df_t.to_parquet(out_p, engine='pyarrow')
        print(f"✅ 文件 {file_name} 已转换至 Parquet.")

# ==========================================
# 模块 2: 自动化核对逻辑
# ==========================================
def verify_random_samples(parquet_path, num_samples=10):
    df = pd.read_parquet(parquet_path)
    all_ids = df['track_id'].unique()
    sampled_ids = np.random.choice(all_ids, size=min(num_samples, len(all_ids)), replace=False)
    
    samples = []
    for tid in sampled_ids:
        track_data = df[df['track_id'] == tid].sort_values('timestamp')
        samples.append({
            'track_id': tid,
            'count': len(track_data),
            'start': track_data['timestamp'].min(),
            'end': track_data['timestamp'].max(),
            'p_first_lat': track_data['lat'].iloc[0]
        })
    return samples

def verify_against_raw(raw_csv_path, sampled_ids):
    verification_results = {}
    with open(raw_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            row = [x.strip() for x in row if x.strip()]
            if not row: continue
            tid = int(row[0])
            if tid in sampled_ids:
                dyn = row[10:]
                verification_results[tid] = {
                    'raw_total_points': len(dyn) // 6,
                    'raw_duration': float(dyn[-1]) - float(dyn[5]),
                    'raw_first_lat': float(dyn[0])
                }
    return verification_results

# ==========================================
# 模块 3: 运行主程序与生成报告
# ==========================================
if __name__ == "__main__":
    # 1. 解析数据
    run_batch_parser()

    # 2. 选取一个文件进行深度核对
    test_file = '20181024_d1_0830_0900' # 修改为你 dataset 下的文件名
    raw_path = f'dataset/{test_file}.csv'
    parquet_path = f'processed_data/{test_file}.parquet'

    if os.path.exists(raw_path) and os.path.exists(parquet_path):
        print(f"\n" + "="*90)
        print(f"📊 随机抽样对比报告: {test_file}")
        print(f"{'Track_ID':<10} | {'点数(处理/原始)':<15} | {'时间差误差(s)':<15} | {'首点坐标':<10} | {'结论'}")
        print("-" * 90)

        samples = verify_random_samples(parquet_path, 10)
        raw_infos = verify_against_raw(raw_path, [s['track_id'] for s in samples])

        for s in samples:
            tid = s['track_id']
            raw = raw_infos.get(tid)
            if not raw: continue

            # 点数判定 (25Hz -> 1Hz)
            theory_count = (raw['raw_total_points'] + 24) // 25
            # 时间跨度判定
            p_dur = (s['end'] - s['start']).total_seconds()
            err = abs(p_dur - raw['raw_duration'])
            # 坐标判定
            coord_ok = abs(s['p_first_lat'] - raw['raw_first_lat']) < 1e-6
            
            status = "✅ PASS" if err < 1.5 and coord_ok else "❌ FAIL"
            
            print(f"{tid:<10} | {s['count']}/{theory_count:<12} | {err:<15.4f} | {'Matched' if coord_ok else 'Shifted':<10} | {status}")
        print("="*90)
    else:
        print("\n无法进行核对，请检查文件名配置。")