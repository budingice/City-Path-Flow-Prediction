import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def mine_typical_congestion():
    # --- 保持你的文件结构 ---
    input_dir = "matched_data"    # 使用匹配后的轨迹点数据
    output_dir = "model_inputs"   # 结果存放位置
    analysis_dir = "analysis_results" # 新增：存放统计分析结果
    
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # 1. 扫描所有匹配后的文件
    matched_files = sorted(glob.glob(os.path.join(input_dir, "*_matched.parquet")))
    if not matched_files:
        print(f"❌ 在 {input_dir} 下未找到数据。")
        return

    print(f"🚀 专家模式启动：正在深度挖掘 {len(matched_files)} 个片段中的拥堵切片...")
    
    all_window_stats = []

    for file_path in tqdm(matched_files):
        df = pd.read_parquet(file_path)
        
        # 确保时间戳格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 定义 60s 时间窗口
        df['time_window'] = df['timestamp'].dt.floor('60s')
        
        # 2. 计算时空特征：流量 (Volume) 和 平均速度 (Speed)
        # 聚合逻辑：按 路段 + 时间窗口
        stats = df.groupby(['edge_id', 'time_window']).agg(
            volume=('track_id', 'nunique'),
            avg_speed=('speed', 'mean')
        ).reset_index()
        
        all_window_stats.append(stats)

    # 合并全局统计量
    full_stats = pd.concat(all_window_stats, ignore_index=True)

    # 3. 识别典型拥堵 (IQR 原则 & 物理指标叠加)
    print("🔍 正在计算全局基准并识别异常...")
    
    # 计算每个路段的自由流速度 (95% 分位数)
    free_flow = full_stats.groupby('edge_id')['avg_speed'].quantile(0.95).rename('v_free')
    full_stats = full_stats.join(free_flow, on='edge_id')

    # 计算 TTI (Travel Time Index)
    full_stats['TTI'] = full_stats['v_free'] / full_stats['avg_speed'].replace(0, np.nan)

    # --- IQR 异常检测逻辑 ---
    # 定义流量激增：Volume > Q3 + 1.5*IQR
    def get_iqr_outliers(group):
        q1 = group['volume'].quantile(0.25)
        q3 = group['volume'].quantile(0.75)
        iqr = q3 - q1
        vol_limit = q3 + 1.5 * iqr
        
        # 典型拥堵：流量异常高 且 速度低于限速的 30% (TTI > 3.33)
        is_congested = (group['volume'] > vol_limit) & (group['avg_speed'] < group['v_free'] * 0.3)
        return group[is_congested]

    # 提取典型拥堵切片
    typical_congestion = full_stats.groupby('edge_id', group_keys=False).apply(get_iqr_outliers)

    # 4. 保存结果，对接 step6 训练器
    full_stats.to_parquet(os.path.join(output_dir, "st_congestion_features.parquet"))
    typical_congestion.to_csv(os.path.join(analysis_dir, "typical_congestion_slices.csv"), index=False)
    
    print(f"\n✨ 挖掘完成！")
    print(f"📊 总数据点: {len(full_stats)}")
    print(f"🚩 识别出典型拥堵切片: {len(typical_congestion)} 个")
    print(f"📂 结果已存入 {analysis_dir}/typical_congestion_slices.csv")

if __name__ == "__main__":
    mine_typical_congestion()