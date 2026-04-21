"""
轨迹去噪质量校验脚本

功能：对比原始匹配数据与清洗后数据，评估数据清洗的合适性

输出：
1. 清洗统计报告（控制台）
2. 可视化对比图表
3. 详细的CSV统计报告
运行：python verify_denoise_quality.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data_pair(date_slot):
    """加载原始和清洗后数据对"""
    raw_path = f'matched_data/{date_slot}_matched.parquet'
    clean_path = f'cleaned_data/{date_slot}_matched_cleaned.parquet'
    
    df_raw = pd.read_parquet(raw_path)
    df_clean = pd.read_parquet(clean_path)
    
    return df_raw, df_clean


def compute_basic_stats(df_raw, df_clean):
    """计算基本清洗统计"""
    raw_cnt = len(df_raw)
    clean_cnt = len(df_clean)
    loss_rate = (raw_cnt - clean_cnt) / raw_cnt * 100 if raw_cnt > 0 else 0
    
    raw_vehicles = df_raw['track_id'].nunique()
    clean_vehicles = df_clean['track_id'].nunique()
    vehicle_loss_rate = (raw_vehicles - clean_vehicles) / raw_vehicles * 100 if raw_vehicles > 0 else 0
    
    return {
        'raw_points': raw_cnt,
        'clean_points': clean_cnt,
        'point_loss_rate': loss_rate,
        'raw_vehicles': raw_vehicles,
        'clean_vehicles': clean_vehicles,
        'vehicle_loss_rate': vehicle_loss_rate,
    }


def compute_speed_stats(df_raw, df_clean):
    """计算速度统计"""
    stats = {
        'raw_speed_mean': df_raw['speed'].mean(),
        'raw_speed_std': df_raw['speed'].std(),
        'raw_speed_max': df_raw['speed'].max(),
        'raw_speed_min': df_raw['speed'].min(),
        'clean_speed_mean': df_clean['speed'].mean(),
        'clean_speed_std': df_clean['speed'].std(),
        'clean_speed_max': df_clean['speed'].max(),
        'clean_speed_min': df_clean['speed'].min(),
        'raw_speed_q95': df_raw['speed'].quantile(0.95),
        'clean_speed_q95': df_clean['speed'].quantile(0.95),
        'raw_anomalies_above_120kmh': (df_raw['speed'] > 33.3).sum(),  # 120 km/h
        'clean_anomalies_above_120kmh': (df_clean['speed'] > 33.3).sum(),
    }
    return stats


def compute_vehicle_retention(df_raw, df_clean):
    """计算各车辆的保留率"""
    raw_points_per_vehicle = df_raw.groupby('track_id').size()
    clean_points_per_vehicle = df_clean.groupby('track_id').size()
    
    # 找到保留过数据的车辆
    retained_vehicles = clean_points_per_vehicle.index
    retention_rates = (clean_points_per_vehicle / raw_points_per_vehicle[retained_vehicles] * 100).to_frame(name='retention_rate')
    retention_rates['vehicle'] = retention_rates.index
    
    return retention_rates


def compute_temporal_coverage(df_raw, df_clean):
    """计算时间覆盖率"""
    raw_time_range = (df_raw['timestamp'].max() - df_raw['timestamp'].min()).total_seconds()
    clean_time_range = (df_clean['timestamp'].max() - df_clean['timestamp'].min()).total_seconds()
    
    return {
        'raw_time_span_seconds': raw_time_range,
        'clean_time_span_seconds': clean_time_range,
        'time_coverage_ratio': clean_time_range / raw_time_range if raw_time_range > 0 else 0,
    }


def create_visualizations(df_raw, df_clean, output_dir, output_prefix='verification'):
    """生成可视化对比图"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 速度分布对比 (KDE)
    ax1 = plt.subplot(2, 3, 1)
    sns.kdeplot(data=df_raw, x='speed', label='原始数据', fill=True, alpha=0.5, ax=ax1)
    sns.kdeplot(data=df_clean, x='speed', label='清洗后数据', fill=True, alpha=0.5, ax=ax1)
    ax1.set_xlabel('速度 (m/s)')
    ax1.set_ylabel('密度')
    ax1.set_title('速度分布密度对比')
    ax1.legend()
    ax1.axvline(x=33.3, color='red', linestyle='--', label='120km/h阈值')
    
    # 2. 速度箱线图
    ax2 = plt.subplot(2, 3, 2)
    speed_data = pd.DataFrame({
        '原始': df_raw['speed'],
        '清洗后': df_clean['speed']
    })
    speed_data.boxplot(ax=ax2)
    ax2.set_ylabel('速度 (m/s)')
    ax2.set_title('速度分布箱线图')
    ax2.grid(True, alpha=0.3)
    
    # 3. 流量趋势对比 (按10秒聚合)
    ax3 = plt.subplot(2, 3, 3)
    flow_raw = df_raw.groupby(pd.Grouper(key='timestamp', freq='10S'))['track_id'].nunique()
    flow_clean = df_clean.groupby(pd.Grouper(key='timestamp', freq='10S'))['track_id'].nunique()
    
    ax3.plot(flow_raw.index, flow_raw.values, alpha=0.6, label='原始流量', marker='o', markersize=3)
    ax3.plot(flow_clean.index, flow_clean.values, alpha=0.8, label='清洗后流量', marker='s', markersize=3)
    ax3.set_xlabel('时间')
    ax3.set_ylabel('活跃车辆数')
    ax3.set_title('流量趋势对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 车辆保留率分布
    ax4 = plt.subplot(2, 3, 4)
    retention_df = compute_vehicle_retention(df_raw, df_clean)
    ax4.hist(retention_df['retention_rate'], bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(retention_df['retention_rate'].mean(), color='red', linestyle='--', label=f"平均: {retention_df['retention_rate'].mean():.1f}%")
    ax4.set_xlabel('保留率 (%)')
    ax4.set_ylabel('车辆数')
    ax4.set_title('车辆点数保留率分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 速度直方图对比
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(df_raw['speed'], bins=50, alpha=0.5, label='原始', edgecolor='black')
    ax5.hist(df_clean['speed'], bins=50, alpha=0.5, label='清洗后', edgecolor='black')
    ax5.axvline(x=33.3, color='red', linestyle='--', linewidth=2, label='速度阈值(33.3m/s)')
    ax5.set_xlabel('速度 (m/s)')
    ax5.set_ylabel('数据点数')
    ax5.set_title('速度分布直方图')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 数据损耗统计
    ax6 = plt.subplot(2, 3, 6)
    categories = ['数据点数', '车辆数']
    raw_vals = [len(df_raw), df_raw['track_id'].nunique()]
    clean_vals = [len(df_clean), df_clean['track_id'].nunique()]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax6.bar(x - width/2, raw_vals, width, label='原始', alpha=0.8)
    ax6.bar(x + width/2, clean_vals, width, label='清洗后', alpha=0.8)
    ax6.set_ylabel('数量')
    ax6.set_title('清洗前后数据规模对比')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(raw_vals, clean_vals)):
        ax6.text(i - width/2, v1, f'{v1}', va='bottom', ha='center', fontsize=9)
        ax6.text(i + width/2, v2, f'{v2}', va='bottom', ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f'{output_prefix}_visualization.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {output_path}")
    
    return fig


def generate_detailed_report(date_slot, output_dir):
    """生成单个时间段的详细报告"""
    
    print(f"\n{'='*70}")
    print(f"校验报告: {date_slot}")
    print(f"{'='*70}\n")
    
    try:
        df_raw, df_clean = load_data_pair(date_slot)
    except Exception as e:
        print(f"❌ 无法加载数据: {e}")
        return None
    
    # 基本统计
    basic_stats = compute_basic_stats(df_raw, df_clean)
    print("【1. 基本清洗统计】")
    print(f"  原始数据点数:        {basic_stats['raw_points']:,}")
    print(f"  清洗后数据点数:      {basic_stats['clean_points']:,}")
    print(f"  数据损耗率:          {basic_stats['point_loss_rate']:.2f}%")
    print(f"  原始车辆数:          {basic_stats['raw_vehicles']}")
    print(f"  清洗后车辆数:        {basic_stats['clean_vehicles']}")
    print(f"  车辆损失率:          {basic_stats['vehicle_loss_rate']:.2f}%")
    
    # 速度统计
    speed_stats = compute_speed_stats(df_raw, df_clean)
    print("\n【2. 速度分布统计】(单位: m/s)")
    print(f"  原始数据:")
    print(f"    - 均值:            {speed_stats['raw_speed_mean']:.2f} m/s ({speed_stats['raw_speed_mean']*3.6:.2f} km/h)")
    print(f"    - 标准差:          {speed_stats['raw_speed_std']:.2f} m/s")
    print(f"    - 最大值:          {speed_stats['raw_speed_max']:.2f} m/s ({speed_stats['raw_speed_max']*3.6:.2f} km/h)")
    print(f"    - 最小值:          {speed_stats['raw_speed_min']:.2f} m/s")
    print(f"    - 95分位数:        {speed_stats['raw_speed_q95']:.2f} m/s")
    print(f"    - >120km/h异常值:  {speed_stats['raw_anomalies_above_120kmh']} 个")
    
    print(f"  清洗后数据:")
    print(f"    - 均值:            {speed_stats['clean_speed_mean']:.2f} m/s ({speed_stats['clean_speed_mean']*3.6:.2f} km/h)")
    print(f"    - 标准差:          {speed_stats['clean_speed_std']:.2f} m/s")
    print(f"    - 最大值:          {speed_stats['clean_speed_max']:.2f} m/s ({speed_stats['clean_speed_max']*3.6:.2f} km/h)")
    print(f"    - 最小值:          {speed_stats['clean_speed_min']:.2f} m/s")
    print(f"    - 95分位数:        {speed_stats['clean_speed_q95']:.2f} m/s")
    print(f"    - >120km/h异常值:  {speed_stats['clean_anomalies_above_120kmh']} 个")
    
    # 时间覆盖
    temporal_stats = compute_temporal_coverage(df_raw, df_clean)
    print(f"\n【3. 时间覆盖统计】")
    print(f"  原始时间跨度:        {temporal_stats['raw_time_span_seconds']:.0f} 秒")
    print(f"  清洗后时间跨度:      {temporal_stats['clean_time_span_seconds']:.0f} 秒")
    print(f"  时间覆盖率:          {temporal_stats['time_coverage_ratio']*100:.2f}%")
    
    # 车辆保留率统计
    retention_df = compute_vehicle_retention(df_raw, df_clean)
    print(f"\n【4. 车辆保留率统计】")
    print(f"  保留车辆数:          {len(retention_df)}")
    print(f"  平均保留率:          {retention_df['retention_rate'].mean():.2f}%")
    print(f"  保留率中位数:        {retention_df['retention_rate'].median():.2f}%")
    print(f"  保留率标准差:        {retention_df['retention_rate'].std():.2f}%")
    print(f"  最低保留率:          {retention_df['retention_rate'].min():.2f}%")
    print(f"  最高保留率:          {retention_df['retention_rate'].max():.2f}%")
    
    # 路段覆盖率
    raw_edges = df_raw['edge_id'].nunique()
    clean_edges = df_clean['edge_id'].nunique()
    print(f"\n【5. 路段覆盖统计】")
    print(f"  原始路段数:          {raw_edges}")
    print(f"  清洗后路段数:        {clean_edges}")
    print(f"  路段保留率:          {clean_edges/raw_edges*100:.2f}%")
    
    # 生成可视化
    print(f"\n【6. 生成可视化对比图...】")
    create_visualizations(df_raw, df_clean, output_dir, date_slot)
    
    # 保存详细统计
    all_stats = {**basic_stats, **speed_stats, **temporal_stats}
    return all_stats, retention_df


def verify_all_slots(output_dir):
    """校验所有时间段"""
    
    matched_dir = Path('matched_data')
    matched_files = sorted(matched_dir.glob('*_matched.parquet'))
    
    date_slots = [f.stem.replace('_matched', '') for f in matched_files]
    
    print(f"\n找到 {len(date_slots)} 个数据时间段")
    
    all_reports = []
    
    for slot in date_slots:
        try:
            report, retention_df = generate_detailed_report(slot, output_dir)
            if report:
                report['date_slot'] = slot
                all_reports.append(report)
                
                # 保存车辆保留率详情
                retention_csv = output_dir / f'{slot}_retention_detail.csv'
                retention_df.to_csv(str(retention_csv), index=False)
        except Exception as e:
            print(f"❌ {slot} 处理失败: {e}")
    
    # 汇总报告
    if all_reports:
        summary_df = pd.DataFrame(all_reports)
        summary_path = output_dir / 'denoise_verification_summary.csv'
        summary_df.to_csv(str(summary_path), index=False)
        print(f"\n✓ 汇总报告已保存: {summary_path}")
        
        print(f"\n{'='*70}")
        print("【汇总统计 - 所有时间段】")
        print(f"{'='*70}")
        print(f"平均点数损耗率:        {summary_df['point_loss_rate'].mean():.2f}%")
        print(f"平均车辆损失率:        {summary_df['vehicle_loss_rate'].mean():.2f}%")
        print(f"平均速度(清洗后):      {summary_df['clean_speed_mean'].mean():.2f} m/s")
        print(f"平均离群值总数(原):    {summary_df['raw_anomalies_above_120kmh'].mean():.0f}")
        print(f"平均离群值总数(清洗后): {summary_df['clean_anomalies_above_120kmh'].mean():.0f}")


def verify_single_slot(date_slot, output_dir=None):
    """校验单个时间段"""
    if output_dir is None:
        output_dir = Path('eda_results/denoise_verification')
        output_dir.mkdir(parents=True, exist_ok=True)
    report, retention_df = generate_detailed_report(date_slot, output_dir)
    return report, retention_df


if __name__ == '__main__':
    # 创建输出目录
    output_dir = Path('eda_results/denoise_verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 校验所有时间段
    verify_all_slots(output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ 校验完成！")
    print(f"✓ 所有结果已保存到: {output_dir}")
    print(f"{'='*70}")
