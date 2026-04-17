import pandas as pd
import os

# 请确保你已经运行过 map_matching() 并生成了文件
matched_file = "matched_data/20181024_d1_0830_0900_matched.parquet" # 或者是你实际的文件名

if os.path.exists(matched_file):
    df = pd.read_parquet(matched_file)
    # 我们找一个数据比较全的车辆 (假设 track_id=1)
    sample = df[df['track_id'] == 1].head(10) 
    
    print("--- 请把下面这段数据贴给我 ---")
    print(sample[['track_id', 'lat', 'lon', 'u', 'v']].to_string(index=False))
else:
    print("⚠️ 还没找到匹配后的文件，请先运行你的 map_matching 脚本。")