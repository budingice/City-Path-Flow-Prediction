import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_path_volatility(df_list, time_labels):
    """基于 edge_id 统计唯一车辆数并分析波动性"""
    all_snaps = []
    for df, label in zip(df_list, time_labels):
        flow_snap = df.groupby('edge_id')['track_id'].nunique().reset_index(name='flow')
        flow_snap['time_label'] = label
        all_snaps.append(flow_snap)

    full_df = pd.concat(all_snaps, ignore_index=True)
    pivot = full_df.pivot_table(index='edge_id', columns='time_label', values='flow', fill_value=0)
    
    stats = pd.DataFrame(index=pivot.index)
    stats['mean_flow'] = pivot.mean(axis=1)
    stats['std_flow'] = pivot.std(axis=1)
    stats['cv'] = stats['std_flow'] / (stats['mean_flow'] + 1e-6)
    
    return pivot, stats

def generate_quality_report(df_raw, df_clean, output_path, title_suffix=""):
    """
    整合后的六宫格校验图
    注意：确保 df_raw 和 df_clean 中包含 'speed' 或 'inst_speed' 列
    """
    # 自动识别速度列名
    s_col = 'speed' if 'speed' in df_clean.columns else 'inst_speed'
    
    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(f"数据去噪质量校验报告 - {title_suffix}", fontsize=20)

    # 1. 速度分布密度对比 (KDE)
    ax1 = plt.subplot(2, 3, 1)
    sns.kdeplot(data=df_raw, x=s_col, label='原始数据', fill=True, alpha=0.5, ax=ax1)
    sns.kdeplot(data=df_clean, x=s_col, label='清洗后数据', fill=True, alpha=0.5, ax=ax1)
    ax1.set_title('速度分布密度对比')
    ax1.axvline(x=33.3, color='red', linestyle='--', label='120km/h阈值')
    ax1.legend()

    # 2. 速度箱线图
    ax2 = plt.subplot(2, 3, 2)
    sns.boxplot(data=[df_raw[s_col], df_clean[s_col]], ax=ax2)
    ax2.set_xticklabels(['原始', '清洗后'])
    ax2.set_title('速度分布箱线图')

    # 3. 流量趋势对比
    ax3 = plt.subplot(2, 3, 3)
    flow_raw = df_raw.groupby(pd.Grouper(key='timestamp', freq='1T'))['track_id'].nunique()
    flow_clean = df_clean.groupby(pd.Grouper(key='timestamp', freq='1T'))['track_id'].nunique()
    ax3.plot(flow_raw.index, flow_raw.values, label='原始流量', alpha=0.6, marker='o', markersize=2)
    ax3.plot(flow_clean.index, flow_clean.values, label='清洗后流量', alpha=0.8, marker='s', markersize=2)
    ax3.set_title('每分钟活跃车辆数对比')
    ax3.legend()

    # 4. 数据损耗统计 (柱状图)
    ax4 = plt.subplot(2, 3, 4)
    categories = ['数据点数', '车辆数']
    raw_vals = [len(df_raw), df_raw['track_id'].nunique()]
    clean_vals = [len(df_clean), df_clean['track_id'].nunique()]
    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, raw_vals, width, label='原始', color='gray', alpha=0.6)
    ax4.bar(x + width/2, clean_vals, width, label='清洗后', color='green', alpha=0.6)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_title('清洗前后规模对比')
    ax4.legend()

    # 5. 变异系数分布 (CV)
    ax5 = plt.subplot(2, 3, 5)
    # 这里我们计算每个轨迹点的采样间隔稳定性
    df_clean['dt'] = df_clean.groupby('track_id')['timestamp'].diff().dt.total_seconds()
    sns.histplot(df_clean['dt'].dropna(), bins=30, ax=ax5, color='orange')
    ax5.set_title('清洗后采样间隔分布(s)')

    # 6. 速度直方图
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(df_raw[s_col], bins=50, alpha=0.4, label='原始')
    ax6.hist(df_clean[s_col], bins=50, alpha=0.4, label='清洗后')
    ax6.set_title('速度区间分布直方图')
    ax6.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_heatmap(df, output_path, center=(37.977, 23.737)):
    """生成热力图"""
    try:
        import folium
        from folium.plugins import HeatMap
        m = folium.Map(location=center, zoom_start=13, tiles='cartodbpositron')
        heat_data = df[['lat', 'lon']].dropna().values.tolist()
        HeatMap(heat_data, radius=8, blur=12).add_to(m)
        m.save(output_path)
    except ImportError:
        logging.warning("未安装 folium，跳过热力图生成")

def plot_path_kinematics_report(path_df, output_dir, label=""):
    """
    针对阶段 6 产出的路径特征进行可视化
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 5))
    plt.suptitle(f"路径运动学特征分析 - {label}", fontsize=16)

    # 1. 路径长度分布 (经过的路段数)
    ax1 = plt.subplot(1, 3, 1)
    sns.countplot(data=path_df, x='path_len', palette='viridis', ax=ax1)
    ax1.set_title('路径长度分布 (路段数)')
    ax1.set_xlabel('路段数量')
    ax1.set_ylabel('路径数量')

    # 2. 路径平均速度与 CV 的关系 (散点图)
    # 观察是否存在“速度越慢，波动越大”的现象
    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(data=path_df, x='avg_speed', y='path_cv', alpha=0.4, color='coral', ax=ax2)
    ax2.set_title('平均速度 vs 变异系数 (CV)')
    ax2.set_xlabel('平均速度 (m/s)')
    ax2.set_ylabel('路径 CV')

    # 3. 路径耗时分布 (秒)
    ax3 = plt.subplot(1, 3, 3)
    sns.histplot(path_df['duration'], bins=30, kde=True, color='seagreen', ax=ax3)
    ax3.set_title('路径总耗时分布 (s)')
    ax3.set_xlabel('耗时 (秒)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{label}_path_analysis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path