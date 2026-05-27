import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from matplotlib import font_manager

# 屏蔽警告
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 字体与格式精细配置 (小五号, 宋体, Times New Roman)
# ---------------------------------------------------------
# 设置全局字体：优先尝试宋体 (SimSun)
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'SimHei', 'serif']
plt.rcParams['font.serif'] = ['Times New Roman', 'serif']
plt.rcParams['axes.unicode_minus'] = False 

# 小五号字对应 9pt
FONT_SIZE = 9
plt.rcParams['font.size'] = FONT_SIZE

def set_axis_style(ax):
    """强制设置：标题/标签为宋体，坐标轴数字为 Times New Roman"""
    # 坐标轴刻度数字使用 Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(FONT_SIZE)
    # 标签与标题
    ax.title.set_fontsize(FONT_SIZE)
    ax.xaxis.label.set_fontsize(FONT_SIZE)
    ax.yaxis.label.set_fontsize(FONT_SIZE)

# ---------------------------------------------------------
# 2. 路径配置 (根据您的截图适配)
# ---------------------------------------------------------
# 原始匹配数据路径 (截图显示在 data/processed/ 下)
RAW_DATA_DIR = Path('data/processed')
# 清洗后数据路径 (根据截图推测在 data/processed/cleaned/ 下)
CLEAN_DATA_DIR = Path('data/processed/cleaned')
# 结果输出路径
OUTPUT_BASE = Path('eda_results/denoise_verification')

# ---------------------------------------------------------
# 3. 绘图重构：拆分独立 SVG 并存入子文件夹
# ---------------------------------------------------------

def create_visualizations(df_raw, df_clean, date_slot):
    """生成6张独立的子图文件"""
    
    # 创建该时间段对应的独立子文件夹
    plot_sub_dir = OUTPUT_BASE / f"{date_slot}_plots"
    plot_sub_dir.mkdir(parents=True, exist_ok=True)
    
    def finalize_and_save(name):
        plt.tight_layout(pad=1.0) 
        save_path = plot_sub_dir / f"{date_slot}_{name}.svg"
        plt.savefig(str(save_path), format='svg', bbox_inches='tight')
        plt.close()

    # --- 图1: 速度分布 KDE ---
    plt.figure(figsize=(5, 4))
    sns.kdeplot(data=df_raw, x='speed', label='原始数据', fill=True, alpha=0.3)
    sns.kdeplot(data=df_clean, x='speed', label='清洗后数据', fill=True, alpha=0.3)
    plt.axvline(x=33.3, color='red', linestyle='--', label='120km/h阈值')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('密度')
    plt.title('速度分布密度对比')
    plt.legend(prop={'family': 'SimSun', 'size': FONT_SIZE})
    set_axis_style(plt.gca())
    finalize_and_save('1_speed_kde')

    # --- 图2: 速度箱线图 ---
    plt.figure(figsize=(5, 4))
    plt.boxplot([df_raw['speed'].dropna(), df_clean['speed'].dropna()], labels=['原始', '清洗后'])
    plt.ylabel('速度 (m/s)')
    plt.title('速度分布箱线图')
    plt.grid(True, axis='y', alpha=0.2)
    set_axis_style(plt.gca())
    finalize_and_save('2_speed_boxplot')

    # --- 图3: 流量趋势 ---
    plt.figure(figsize=(8, 4))
    flow_raw = df_raw.groupby(pd.Grouper(key='timestamp', freq='10S'))['track_id'].nunique()
    flow_clean = df_clean.groupby(pd.Grouper(key='timestamp', freq='10S'))['track_id'].nunique()
    plt.plot(flow_raw.index, flow_raw.values, label='原始', marker='o', markersize=2, alpha=0.5)
    plt.plot(flow_clean.index, flow_clean.values, label='清洗后', marker='s', markersize=2, alpha=0.8)
    plt.xlabel('时间')
    plt.ylabel('活跃车辆数')
    plt.title('流量趋势对比')
    plt.legend(prop={'family': 'SimSun', 'size': FONT_SIZE})
    set_axis_style(plt.gca())
    finalize_and_save('3_flow_trend')

    # --- 图4: 车辆保留率分布 ---
    plt.figure(figsize=(5, 4))
    raw_p = df_raw.groupby('track_id').size()
    clean_p = df_clean.groupby('track_id').size()
    common_idx = clean_p.index.intersection(raw_p.index)
    ret_rate = (clean_p[common_idx] / raw_p[common_idx] * 100)
    plt.hist(ret_rate, bins=30, edgecolor='white', color='#5DADE2')
    plt.xlabel('保留率 (%)')
    plt.ylabel('车辆数')
    plt.title('车辆点数保留率分布')
    set_axis_style(plt.gca())
    finalize_and_save('4_retention_hist')

    # --- 图5: 速度直方图 ---
    plt.figure(figsize=(5, 4))
    plt.hist(df_raw['speed'], bins=50, alpha=0.4, label='原始', color='gray')
    plt.hist(df_clean['speed'], bins=50, alpha=0.5, label='清洗后', color='blue')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('频数')
    plt.title('速度分布直方图对比')
    plt.legend(prop={'family': 'SimSun', 'size': FONT_SIZE})
    set_axis_style(plt.gca())
    finalize_and_save('5_speed_hist')

    # --- 图6: 数据规模对比 ---
    plt.figure(figsize=(5, 4))
    labels = ['数据点', '车辆数']
    rv = [len(df_raw), df_raw['track_id'].nunique()]
    cv = [len(df_clean), df_clean['track_id'].nunique()]
    x = np.arange(len(labels))
    plt.bar(x - 0.2, rv, 0.4, label='原始', color='#AED6F1')
    plt.bar(x + 0.2, cv, 0.4, label='清洗后', color='#3498DB')
    plt.xticks(x, labels)
    plt.ylabel('数量')
    plt.title('清洗前后数据规模对比')
    plt.legend(prop={'family': 'SimSun', 'size': FONT_SIZE})
    for i, val in enumerate(rv):
        plt.text(i - 0.2, val, str(val), ha='center', va='bottom', fontname='Times New Roman')
    for i, val in enumerate(cv):
        plt.text(i + 0.2, val, str(val), ha='center', va='bottom', fontname='Times New Roman')
    set_axis_style(plt.gca())
    finalize_and_save('6_scale_bar')

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------

def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # 扫描 data/processed/ 下所有的 matched.parquet 文件
    matched_files = sorted(RAW_DATA_DIR.glob('*_matched.parquet'))
    
    if not matched_files:
        print(f"❌ 错误: 在 {RAW_DATA_DIR} 未找到匹配文件。请确认路径！")
        return

    print(f"🚀 找到 {len(matched_files)} 个文件，开始处理...")

    for f_path in matched_files:
        date_slot = f_path.stem.replace('_matched', '')
        # 对应清洗后的文件名（假设在 cleaned 目录下多了一个后缀或同名）
        clean_file = CLEAN_DATA_DIR / f"{date_slot}_matched_cleaned.parquet"
        
        if not clean_file.exists():
            print(f"⚠️ 跳过 {date_slot}: 找不到对应的清洗后文件 {clean_file}")
            continue
            
        print(f"正在分析: {date_slot}...")
        try:
            df_raw = pd.read_parquet(f_path)
            df_clean = pd.read_parquet(clean_file)
            
            # 确保时间戳列为 datetime 类型
            for df in [df_raw, df_clean]:
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            create_visualizations(df_raw, df_clean, date_slot)
            print(f"  ✅ 成功：图表已存至 {OUTPUT_BASE}/{date_slot}_plots/")
            
        except Exception as e:
            print(f"  ❌ 出错：{date_slot} -> {e}")

if __name__ == '__main__':
    main()