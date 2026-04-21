import pandas as pd
import os
import glob

def check_matched_results():
    matched_dir = "matched_data"
    # 获取该目录下所有的匹配结果文件
    matched_files = glob.glob(os.path.join(matched_dir, "*_matched.parquet"))
    
    if not matched_files:
        print(f"❌ 在 {matched_dir} 文件夹中没有找到任何结果文件。请确认 Step 3 运行成功。")
        return

    # 读取第一个匹配好的文件进行查看
    sample_file = matched_files[0]
    print(f"📄 正在查看样本文件: {os.path.basename(sample_file)}")
    
    df = pd.read_parquet(sample_file)
    
    # 打印前 10 行，查看新增的列
    print("\n--- 数据预览 (前10行) ---")
    # 重点关注 track_id, timestamp, u, v, edge_id
    cols_to_show = ['track_id', 'timestamp', 'lat', 'lon', 'u', 'v', 'edge_id']
    # 过滤一下存在的列进行展示
    existing_cols = [c for c in cols_to_show if c in df.columns]
    print(df[existing_cols].head(10))
    
    # 打印统计信息
    print("\n--- 数据统计 ---")
    print(f"总数据行数: {len(df)}")
    print(f"包含的唯一车辆数 (track_id): {df['track_id'].nunique()}")
    print(f"匹配到的唯一路段数 (edge_id): {df['edge_id'].nunique() if 'edge_id' in df.columns else '未找到edge_id'}")
    
    # 检查是否有空值
    if df['u'].isnull().any():
        print("⚠️ 警告：部分点未能匹配到最近的路段（u/v 列包含空值）。")
    else:
        print("✅ 所有轨迹点均成功匹配到路网。")

if __name__ == "__main__":
    check_matched_results()
    