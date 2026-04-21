"""
view_path_kinematics.py
查看 step4_extract_path.py 提取的路径运动学数据

功能:
- 随机查看10条路径的聚合统计信息（平均速度、变异系数、持续时间等）
- 显示每条路径对应的原始轨迹数据（前10行 + 最后5行）
- 自动匹配 path_data 和 matched_data 中的文件

方法:
- 从 path_data 读取 kinematics.parquet 文件获取路径聚合数据
- 从 matched_data 读取对应 matched.parquet 文件获取原始轨迹
- 按 track_id 关联聚合统计和原始数据

用法示例:
    python view_path_kinematics.py
"""

import pandas as pd
import glob
import os

def view_path_kinematics_sample():
    # 1. 自动定位到一个带有运动学指标的文件
    path_files = glob.glob("path_data/*kinematics.parquet")
    if not path_files:
        print("未发现 _path_kinematics 文件，请检查路径。")
        return

    sample_file = path_files[0]  # 读取第一个文件作为样本
    print(f"📂 正在检查文件: {os.path.basename(sample_file)}")

    # 推断对应的 matched_data 文件名
    matched_file = sample_file.replace("_path_kinematics.parquet", "_matched.parquet").replace("path_data", "matched_data")

    if not os.path.exists(matched_file):
        print(f"❌ 对应的原始数据文件不存在: {matched_file}")
        return

    # 读取聚合数据
    kinematics_df = pd.read_parquet(sample_file)
    # 读取原始数据
    matched_df = pd.read_parquet(matched_file)

    # 2. 随机抽取 10 条路径
    if len(kinematics_df) < 10:
        check_list = kinematics_df
        print(f"⚠️ 文件中只有 {len(kinematics_df)} 条路径，将显示全部。")
    else:
        check_list = kinematics_df.sample(10, random_state=42)

    # 3. 格式化输出聚合统计
    print(f"\n--- 随机 {len(check_list)} 条路径深度质检 ---")
    columns_to_show = ['track_id', 'path_len', 'avg_speed', 'path_cv', 'duration', 'path_signature']
    display_df = check_list[columns_to_show].copy()
    display_df['path_signature'] = display_df['path_signature'].str[:50] + "..."

    print(display_df.to_string(index=False))

    # 4. 显示每条路径的原始数据行
    print(f"\n--- 原始轨迹数据详情 ---")
    for idx, row in check_list.iterrows():
        track_id = row['track_id']
        print(f"\n🚗 车辆 {track_id} 的原始轨迹 (共 {row['point_count']} 个点):")
        track_data = matched_df[matched_df['track_id'] == track_id].sort_values('timestamp')
        # 显示前10行和最后5行，如果太多
        if len(track_data) <= 15:
            print(track_data.to_string(index=False))
        else:
            print("前10行:")
            print(track_data.head(10).to_string(index=False))
            print("...")
            print("最后5行:")
            print(track_data.tail(5).to_string(index=False))

        print("-" * 80)

if __name__ == "__main__":
    view_path_kinematics_sample()