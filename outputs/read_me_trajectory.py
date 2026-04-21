import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 配置界面显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_pneuma_dataset(dataset_dir, topology_report_path):
    print(f"📂 正在扫描目录: {dataset_dir}")
    
    # 获取目录下所有csv文件
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        print("❌ 未发现CSV文件，请检查路径。")
        return

    all_frames = []
    
    # 1. 遍历读取所有文件
    print("⏳ 正在读取并汇总轨迹数据...")
    for file in csv_files:
        # pNEUMA 原始文件通常以分号分隔，且第一行是列名，之后是数据
        # 如果你的文件已经清洗过，直接 pd.read_csv(file) 即可
        try:
            # 自动跳过可能存在的元数据行（视具体数据清洗情况而定）
            temp_df = pd.read_csv(file, sep=None, engine='python')
            all_frames.append(temp_df[['track_id', 'type', 'time']])
        except Exception as e:
            print(f"⚠️ 读取文件 {os.path.basename(file)} 失败: {e}")

    df = pd.concat(all_frames, ignore_index=True)

    # 2. 轨迹统计逻辑
    total_trajectories = df['track_id'].nunique()
    vehicle_types = df['type'].value_counts()
    
    # 3. 随时间分布 (以 10 秒为一档统计在场车辆，曲线会更平滑)
    df['time_bin'] = (df['time'] // 10).astype(int)
    time_distribution = df.groupby('time_bin')['track_id'].nunique()

    # 4. 获取路网信息 (结合你之前汇报的数据)
    # 这里的街道名称是根据你之前的运行结果提取的
    sample_streets = ["Σπύρου Μερκούρη", "Βασιλίσσης Σοφίας", "Βασιλέως Κωνσταντίνου", "Ασκληπιού", "Ριζάρη"]

    # --- 输出可视化报告 ---
    print("\n" + "★"*20 + " 数据汇总报告 " + "★"*20)
    print(f"【数据文件】: 已处理 {len(csv_files)} 个时间段文件")
    print(f"【轨迹规模】: 总计 ID 数 {total_trajectories:,}")
    print(f"【车辆类型】:\n{vehicle_types.to_string()}")
    print("-" * 54)
    print(f"【路网概况】:")
    print(f"  - 街道总数: 309条")
    print(f"  - 拓扑路段: 1184段")
    print(f"  - 交叉口数: 608个")
    print(f"  - 典型路段: {', '.join(sample_streets)}")
    print("★"*54)

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 轨迹类别占比饼图
    vehicle_types.plot(kind='pie', ax=ax1, autopct='%1.1f%%', cmap='viridis')
    ax1.set_title("pNEUMA 轨迹类别组成", fontsize=14)
    ax1.set_ylabel("")

    # 时间分布折线图
    time_distribution.plot(kind='line', ax=ax2, color='#2c3e50', linewidth=1.5)
    ax2.set_title("全时段在场车辆数波动", fontsize=14)
    ax2.set_xlabel("时间刻度 (每10秒)", fontsize=12)
    ax2.set_ylabel("在场车辆 (IDs)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# 根据你的 VS Code 截图，路径应为：
dataset_path = "Dataset"  # 如果脚本在 '路径流量预测' 下运行
report_path = "analysis_results/road_topology_report.csv"

analyze_pneuma_dataset(dataset_path, report_path)