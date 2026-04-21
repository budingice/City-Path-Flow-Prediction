"""
analyze_path_competition.py
高级路径竞争分析脚本 - 针对 pNEUMA 数据集结构化缺失优化版

功能:
1. 时间窗对齐：严格锁定 08:30 - 11:00 观测时段。
2. 局部归一化：实现 [路径流量 / 时段OD总流量]，解决份额稀释问题。
3. 全局统计：分析路网 OD 对总数及路径数量分布占比。
4. 日间异质性分析：按日期对比相同时刻下的路径选择稳定性。

终端调用示例:
    python analyze_path_competition.py --top-n-od 10 --min-flow 3
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import ast
import argparse
from datetime import datetime

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess(input_dir="path_data"):
    """加载数据并执行严格的时间窗口裁剪"""
    files = sorted(glob.glob(os.path.join(input_dir, "*_path_kinematics.parquet")))
    if not files:
        raise FileNotFoundError("未发现数据文件，请检查 path_data 目录")

    print(f"📂 正在加载 {len(files)} 个轨迹文件...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['date'] = df['start_time'].dt.date
    
    # 裁剪 08:30 - 11:00 观测窗
    df = df[(df['start_time'].dt.time >= datetime.strptime("08:30", "%H:%M").time()) & 
            (df['start_time'].dt.time <= datetime.strptime("11:00", "%H:%M").time())]
    
    print(f"✅ 窗口裁剪完成，保留有效观测记录: {len(df)} 条")
    return df

def extract_od_groups(df):
    """解析路径序列并提取 OD 对"""
    def parse_seq(x):
        if isinstance(x, (list, np.ndarray)): return x
        try: return ast.literal_eval(str(x).replace('\x00', ''))
        except: return None

    df['path_sequence'] = df['path_sequence'].apply(parse_seq)
    df = df.dropna(subset=['path_sequence'])
    df = df[df['path_sequence'].map(len) > 0]
    
    df['origin'] = df['path_sequence'].apply(lambda x: x[0])
    df['destination'] = df['path_sequence'].apply(lambda x: x[-1])
    df['od_pair'] = df['origin'].astype(str) + " -> " + df['destination'].astype(str)
    return df

def plot_global_od_stats(df, output_dir):
    """统计功能：对所有 OD 对及其路径数量作统计分析"""
    print("📊 正在生成全局 OD 统计分析...")
    
    # 每个 OD 对拥有的唯一路径数量
    od_path_counts = df.groupby('od_pair')['path_signature'].nunique().reset_index()
    od_path_counts.columns = ['od_pair', 'path_count']
    
    total_ods = len(od_path_counts)
    
    # 统计路径数量的分布情况
    dist = od_path_counts['path_count'].value_counts(normalize=True).sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(dist.index.astype(str), dist.values, color='skyblue', edgecolor='black', alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.title(f'路网 OD 对路径数量分布 (总 OD 数: {total_ods})', fontsize=14)
    plt.xlabel('每个 OD 对拥有的独立路径数量', fontsize=12)
    plt.ylabel('OD 对占比', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "global_od_path_distribution.png"), dpi=300)
    plt.close()
    
    # 保存明细
    od_path_counts.to_csv(os.path.join(output_dir, "od_path_counts_summary.csv"), index=False)

def analyze_local_shares(df, time_bin='15min', min_flow=3):
    """针对结构化缺失优化的局部归一化算法"""
    # 统一时刻轴
    df['time_slot'] = df['start_time'].dt.floor(time_bin).apply(lambda x: x.replace(year=1900, month=1, day=1))

    # 执行透视表计数
    pivot = df.groupby(['date', 'time_slot', 'od_pair', 'path_signature']).size().unstack(fill_value=0)
    
    # 计算每个片段的局部总流量
    row_sums = pivot.sum(axis=1)
    
    # 关键过滤：流量太小不具统计意义
    pivot = pivot[row_sums >= min_flow]
    row_sums = row_sums[row_sums >= min_flow]

    # 执行除法，得到 0-1 之间的份额
    shares = pivot.div(row_sums, axis=0)
    
    return shares.reset_index(), row_sums.reset_index(name='interval_volume')

def plot_daily_comparison(shares_df, volume_df, target_od, output_dir):
    """绘制分时段流量占比可视化展示"""
    od_shares = shares_df[shares_df['od_pair'] == target_od]
    od_volumes = volume_df[volume_df['od_pair'] == target_od]
    
    dates = od_shares['date'].unique()
    if len(dates) == 0: return

    fig, axes = plt.subplots(len(dates), 1, figsize=(12, 4 * len(dates)), sharex=True)
    if len(dates) == 1: axes = [axes]

    for i, date in enumerate(dates):
        day_shares = od_shares[od_shares['date'] == date].sort_values('time_slot')
        day_vols = od_volumes[od_volumes['date'] == date]
        
        # 识别路径列
        path_cols = [c for c in day_shares.columns if c not in ['date', 'time_slot', 'od_pair']]
        
        # 按照路径总流量降序排列，只画前 3 条最重要的路径
        path_rank = day_shares[path_cols].sum().sort_values(ascending=False).index
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for idx, p_col in enumerate(path_rank[:3]):
            axes[i].plot(day_shares['time_slot'], day_shares[p_col], 
                         label=f'路径 {idx+1} 占比', marker='o', 
                         color=colors[idx] if idx < len(colors) else None, linewidth=2)

        axes[i].set_title(f"日期: {date} (时段样本量: {day_vols['interval_volume'].sum()} 辆)", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_ylabel("份额 (0.0-1.0)")
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].legend(loc='upper right', fontsize='small', frameon=True)

    plt.xlabel("高峰时段 (08:30 - 11:00)")
    plt.tight_layout()
    
    safe_fn = target_od.replace(' -> ', '_').replace(' ', '').replace(':', '')
    plt.savefig(os.path.join(output_dir, f"dynamic_share_{safe_fn}.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="pNEUMA Path Competition Analysis")
    parser.add_argument('--top-n-od', type=int, default=10, help='分析流量最大的 N 个 OD 对')
    parser.add_argument('--min-flow', type=int, default=3, help='过滤低流量样本')
    args = parser.parse_args()

    # 初始化输出目录
    out_dir = f"analysis_results/path_competition_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 数据加载与 OD 提取
    raw_df = load_and_preprocess()
    df_with_od = extract_od_groups(raw_df)
    
    # 2. 全局统计分析 (新增功能)
    plot_global_od_stats(df_with_od, out_dir)
    
    # 3. 局部份额计算
    shares_df, volume_df = analyze_local_shares(df_with_od, min_flow=args.min_flow)
    
    # 4. 识别竞争激烈的 Top OD
    top_ods = volume_df.groupby('od_pair')['interval_volume'].sum().nlargest(args.top_n_od).index
    
    # 5. 分时段可视化展示
    print(f"\n📈 正在生成 Top {args.top_n_od} OD 对的动态竞争图表...")
    for od in top_ods:
        plot_daily_comparison(shares_df, volume_df, od, out_dir)
        
    print(f"\n✅ 分析任务完成！请查看目录: {out_dir}")

if __name__ == "__main__":
    main()