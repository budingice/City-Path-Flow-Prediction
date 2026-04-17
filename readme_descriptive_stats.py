import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# --- 环境配置 ---
plt.style.use('ggplot') # 使用更现代的绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def run_descriptive_analysis(input_folder='processed_data', save_folder='analysis_results'):
    # 1. 创建保存结果的文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"📁 已创建结果保存文件夹: {save_folder}")

    # 2. 加载数据
    info_files = glob.glob(os.path.join(input_folder, "*_info.parquet"))
    traj_files = glob.glob(os.path.join(input_folder, "[!_]*[!.info].parquet"))
    
    if not info_files or not traj_files:
        print("❌ 未找到处理后的数据文件，请检查 processed_data 文件夹。")
        return

    # 合并所有车辆基础信息
    df_info = pd.concat([pd.read_parquet(f) for f in info_files]).reset_index(drop=True)
    # 读取第一个轨迹文件作为样本进行深度分析（避免内存溢出）
    df_traj = pd.read_parquet(traj_files[0])
    df_traj['timestamp'] = pd.to_datetime(df_traj['timestamp'])

    print(f"📊 正在生成统计分析报告，样本轨迹点数: {len(df_traj)}...")

    # --- 任务 1: 车型构成分析 ---
    plt.figure(figsize=(10, 6))
    type_counts = df_info['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, palette='magma', legend=False)
    plt.title('pNEUMA 车辆类型分布统计', fontsize=14)
    plt.xlabel('车型', fontsize=12)
    plt.ylabel('车辆数量', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '01_vehicle_type_dist.png'), dpi=300)
    plt.close()

    # --- 任务 2: 速度分布特征 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df_traj['speed'], bins=60, kde=True, color='royalblue')
    plt.title('交通流瞬时速度分布 (Sampling: 1Hz)', fontsize=14)
    plt.xlabel('速度 (km/h)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '02_speed_distribution.png'), dpi=300)
    plt.close()

    # --- 任务 3: 交通流量时间序列 (以1分钟为单位) ---
    flow_min = df_traj.set_index('timestamp').resample('1T')['track_id'].nunique()
    plt.figure(figsize=(12, 6))
    flow_min.plot(kind='line', color='darkgreen', linewidth=2, marker='s', markersize=4)
    plt.fill_between(flow_min.index, flow_min.values, alpha=0.2, color='darkgreen')
    plt.title('监测时段内实时在线车辆数 (Traffic Density)', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('车辆数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '03_flow_time_series.png'), dpi=300)
    plt.close()

    # --- 任务 4: 空间轨迹热力分布 (路网结构预览) ---
    plt.figure(figsize=(8, 10))
    # 使用透明度 alpha 表现密度
    plt.scatter(df_traj['lon'], df_traj['lat'], s=0.05, alpha=0.2, c='orangered')
    plt.title('轨迹空间分布图 (路网拓扑还原)', fontsize=14)
    plt.xlabel('经度', fontsize=12)
    plt.ylabel('纬度', fontsize=12)
    plt.axis('equal') # 保证经纬度比例不失真
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '04_spatial_distribution.png'), dpi=300)
    plt.close()

    # --- 任务 5: 导出数值摘要表格 ---
    summary = df_info.groupby('type')['avg_speed'].agg(['count', 'mean', 'std', 'min', 'max'])
    summary.to_csv(os.path.join(save_folder, 'data_summary_report.csv'))
    print(f"分析区域路网规模：")
    print(f"- 共有 {G.number_of_nodes()} 个拓扑节点")
    print(f"- 共有 {G.number_of_edges()} 条物理路段")
    print(f"- 核心交通交叉口（三岔及以上）共计: {stats['intersection_count']} 处")
    print(f"✨ 分析完成！所有图表已保存至: {os.path.abspath(save_folder)}")

if __name__ == "__main__":
    run_descriptive_analysis()
    