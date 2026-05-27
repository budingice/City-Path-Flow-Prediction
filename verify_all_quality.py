import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 环境与参考图配色配置 (统一采用青色系)
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimSun'] 
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10.5

# 配色方案
COLOR_CYAN = '#8ECFC9'     # 主题青色 (用于原始流量、分布柱状图)
COLOR_ORANGE = '#FA7F6F'   # 珊瑚橙 (用于清洗后流量)
COLOR_LINE = '#5D6D7E'     # 拟合曲线颜色 (深灰蓝)
COLOR_GRID = '#F2F2F2'     # 极浅灰网格

SAVE_DIR = Path('eda_results/final_svg')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def set_academic_axis(ax):
    """移除冗余边框并设置坐标轴字体"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')

# ---------------------------------------------------------
# 2. 数据载入逻辑 (保持不变)
# ---------------------------------------------------------
def load_data_for_final_plots():
    RAW_DIR = Path('data/processed')
    CLEAN_DIR = Path('data/processed/cleaned')
    raw_files = sorted(RAW_DIR.glob('*_matched.parquet'))
    
    all_raw_list, all_clean_list, flow_recs = [], [], []
    
    print("📊 正在全量处理轨迹数据...")
    for f_raw in tqdm(raw_files):
        slot = f_raw.stem.replace('_matched', '')
        f_clean = CLEAN_DIR / f"{slot}_matched_cleaned.parquet"
        if not f_clean.exists(): f_clean = CLEAN_DIR / f"{slot}_matched_clean.parquet"
        
        if f_clean.exists():
            df_r = pd.read_parquet(f_raw)
            df_c = pd.read_parquet(f_clean)
            
            for df in [df_r, df_c]:
                df['norm_time'] = df['timestamp'].apply(lambda x: x.replace(year=1900, month=1, day=1))
            
            r_avg = df_r.groupby(pd.Grouper(key='norm_time', freq='1min'))['track_id'].nunique()
            c_avg = df_c.groupby(pd.Grouper(key='norm_time', freq='1min'))['track_id'].nunique()
            flow_recs.append(pd.concat([r_avg, c_avg], axis=1, keys=['raw', 'clean']))
            
            all_raw_list.append(df_r)
            all_clean_list.append(df_c)

    df_flow = pd.concat(flow_recs).groupby(level=0).mean().sort_index()
    return pd.concat(all_raw_list), pd.concat(all_clean_list), df_flow

# ---------------------------------------------------------
# 3. 绘图执行函数
# ---------------------------------------------------------

def draw_fig3_flow(df_flow):
    """绘制流量对比图：青色原始 vs 橙色清洗"""
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    
    # 原始平均流量：青色虚线
    plt.plot(df_flow.index, df_flow['raw'], color=COLOR_CYAN, label='原始平均流量', 
             linewidth=1.5, linestyle='--', alpha=0.8)
    
    # 清洗后有效流量：橙色实线
    plt.plot(df_flow.index, df_flow['clean'], color=COLOR_ORANGE, label='清洗后有效流量', 
             linewidth=2.2)
    
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.xlabel('典型采样时刻 (H:M)')
    plt.ylabel('平均每分钟活跃车辆数')
    plt.legend(prop={'family': 'SimSun', 'size': 9.5}, loc='upper left', frameon=False)
    plt.grid(axis='y', color=COLOR_GRID, linestyle='-', linewidth=0.6)
    
    set_academic_axis(ax)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig3_flow_cyan_orange.svg", format='svg')
    plt.close()

def draw_fig5_retention(df_raw, df_clean):
    """绘制保留率分布图：全青色柱状图 + 无标签拟合曲线"""
    # 计算保留率
    r_pts = df_raw.groupby('track_id').size()
    c_pts = df_clean.groupby('track_id').size()
    common = c_pts.index.intersection(r_pts.index)
    retention = (c_pts[common] / r_pts[common]) * 100

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    # 1. 柱状图：全部统一为青色
    sns.histplot(retention, 
                 bins=40, 
                 stat="percent", 
                 color=COLOR_CYAN, 
                 alpha=0.5, 
                 edgecolor='white', 
                 label='分布占比')

    # 2. 拟合曲线：移除拟合标签，使用深色增强对比
    sns.kdeplot(retention, 
                color=COLOR_LINE, 
                linewidth=2, 
                label='_nolegend_') # 设置为 _nolegend_ 从而不显示在图例中

    plt.xlabel('轨迹点保留率 (%)')
    plt.ylabel('样本占比 (%)')
    plt.xlim(0, 105)
    plt.ylim(0, None)
    
    # 仅保留分布占比的图例，或根据需求完全移除
    plt.legend(prop={'family': 'SimSun', 'size': 9}, frameon=False)
    
    plt.grid(axis='y', color=COLOR_GRID, linestyle='--', linewidth=0.5)
    set_academic_axis(ax)
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig5_retention_all_cyan.svg", format='svg')
    plt.close()

# ---------------------------------------------------------
# 4. 主程序运行
# ---------------------------------------------------------
if __name__ == '__main__':
    try:
        raw, clean, flow = load_data_for_final_plots()
        
        print("🎨 正在生成论文最终图表 (青色主题)...")
        draw_fig3_flow(flow)
        draw_fig5_retention(raw, clean)
        
        print(f"✨ 矢量图已保存至: {SAVE_DIR}")
    except Exception as e:
        print(f"❌ 运行失败: {e}")