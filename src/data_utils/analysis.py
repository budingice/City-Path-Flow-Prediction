import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ==========================================
# 论文格式全局设置
# ==========================================
plt.rcParams.update({
    'font.sans-serif': ['SimHei'],     # 中文支持
    'font.serif': ['Times New Roman'], # 英文支持
    'axes.unicode_minus': False,       # 解决负号显示
    'font.size': 11,                   # 对应论文小四号/五号字
    'savefig.dpi': 300,                # 高清打印分辨率
    'savefig.format': 'png',           # 默认格式
    'axes.titlesize': 12,
    'axes.labelsize': 11
})

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
    修改说明：
    1. 拆分原有的六宫格，每个指标独立输出一张图
    2. 删除图中标题（plt.title/suptitle）
    3. 符合论文排版：去除上右边框，设置高DPI
    """
    # 自动识别速度列名
    s_col = 'speed' if 'speed' in df_clean.columns else 'inst_speed'
    
    # 基础路径处理：将 output_path 的扩展名去掉，作为子图文件名前缀
    base_name = os.path.splitext(output_path)[0]
    
    # --- 指标 1: 速度分布密度对比 (KDE) ---
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df_raw, x=s_col, label='原始数据', fill=True, alpha=0.4)
    sns.kdeplot(data=df_clean, x=s_col, label='清洗后数据', fill=True, alpha=0.4)
    plt.axvline(x=33.3, color='red', linestyle='--', label='120km/h阈值', linewidth=1)
    plt.xlabel('速度 (m/s)')
    plt.ylabel('密度')
    plt.legend(frameon=False)
    sns.despine() # 移除上右边框
    plt.tight_layout()
    plt.savefig(f"{base_name}_kde.png")
    plt.close()

    # --- 指标 2: 速度箱线图 ---
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=[df_raw[s_col], df_clean[s_col]], palette="Set2", width=0.5)
    plt.xticks([0, 1], ['原始数据', '清洗后数据'])
    plt.ylabel('速度 (m/s)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_name}_boxplot.png")
    plt.close()

    # --- 指标 3: 流量趋势对比 ---
    plt.figure(figsize=(6, 4))
    flow_raw = df_raw.groupby(pd.Grouper(key='timestamp', freq='1T'))['track_id'].nunique()
    flow_clean = df_clean.groupby(pd.Grouper(key='timestamp', freq='1T'))['track_id'].nunique()
    plt.plot(flow_raw.index, flow_raw.values, label='原始流量', alpha=0.5, marker='o', markersize=3, linewidth=1)
    plt.plot(flow_clean.index, flow_clean.values, label='清洗后流量', alpha=0.8, marker='s', markersize=3, linewidth=1)
    plt.xlabel('观测时间')
    plt.ylabel('活跃车辆数 (veh/min)')
    plt.legend(frameon=False)
    plt.xticks(rotation=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_name}_flow_trend.png")
    plt.close()

    # --- 指标 4: 数据损耗统计 (柱状图) ---
    plt.figure(figsize=(6, 4))
    categories = ['数据点数', '车辆总数']
    raw_vals = [len(df_raw), df_raw['track_id'].nunique()]
    clean_vals = [len(df_clean), df_clean['track_id'].nunique()]
    x = np.arange(len(categories))
    width = 0.3
    plt.bar(x - width/2, raw_vals, width, label='原始', color='#A9A9A9', edgecolor='black', alpha=0.8)
    plt.bar(x + width/2, clean_vals, width, label='清洗后', color='#4682B4', edgecolor='black', alpha=0.8)
    plt.xticks(x, categories)
    plt.ylabel('计数')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_name}_loss_stats.png")
    plt.close()

    # --- 指标 5: 清洗后采样间隔分布 ---
    plt.figure(figsize=(6, 4))
    df_clean['dt'] = df_clean.groupby('track_id')['timestamp'].diff().dt.total_seconds()
    sns.histplot(df_clean['dt'].dropna(), bins=30, color='orange', edgecolor='white')
    plt.xlabel('采样间隔 (s)')
    plt.ylabel('频率')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_name}_dt_dist.png")
    plt.close()

    # --- 指标 6: 速度区间直方图 ---
    plt.figure(figsize=(6, 4))
    plt.hist(df_raw[s_col], bins=50, alpha=0.4, label='原始', edgecolor='gray')
    plt.hist(df_clean[s_col], bins=50, alpha=0.4, label='清洗后', edgecolor='gray')
    plt.xlabel('速度区间 (m/s)')
    plt.ylabel('频数')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_name}_speed_hist.png")
    plt.close()

def generate_heatmap(df, output_path, center=(37.977, 23.737)):
    """生成热力图（保持原有 folium 逻辑不变）"""
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
    针对路径运动学特征分析进行图表拆分输出
    """
    os.makedirs(output_dir, exist_ok=True)
    base_save_path = os.path.join(output_dir, f"{label}")

    # 1. 路径长度分布 (Bar)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=path_df, x='path_len', palette='Blues_d', edgecolor='black')
    plt.xlabel('路段数量 (links)')
    plt.ylabel('路径样本数')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_save_path}_len_dist.png")
    plt.close()

    # 2. 路径平均速度与 CV 的关系 (Scatter)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=path_df, x='avg_speed', y='path_cv', alpha=0.5, color='coral', s=20)
    plt.xlabel('平均速度 (m/s)')
    plt.ylabel('变异系数 (CV)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_save_path}_speed_cv_scatter.png")
    plt.close()

    # 3. 路径耗时分布 (Hist)
    plt.figure(figsize=(6, 4))
    sns.histplot(path_df['duration'], bins=30, kde=True, color='seagreen')
    plt.xlabel('路径总耗时 (s)')
    plt.ylabel('频率')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{base_save_path}_duration_dist.png")
    plt.close()

    return base_save_path # 返回路径前缀