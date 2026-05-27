import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 论文排版配置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong']  # 中文宋体
plt.rcParams['font.serif'] = ['Times New Roman']       # 英文数字
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 10.5  # 对应五号字

RAW_DIR = Path('data/processed')
CLEAN_DIR = Path('data/processed/cleaned')
SAVE_DIR = Path('eda_results/loss_analysis')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def set_axis_style(ax):
    """设置学术坐标轴样式"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    # 隐藏上方和右侧的边框，使图形更简洁
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ---------------------------------------------------------
# 2. 数据处理逻辑
# ---------------------------------------------------------
def calculate_metrics():
    raw_files = sorted(RAW_DIR.glob('*_matched.parquet'))
    stats = []
    total_raw, total_clean = 0, 0

    print("🚀 正在扫描文件并计算损耗率...")
    for f_raw in tqdm(raw_files):
        date_slot = f_raw.stem.replace('_matched', '')
        # 兼容 clean 或 cleaned 后缀
        f_clean = CLEAN_DIR / f"{date_slot}_matched_cleaned.parquet"
        if not f_clean.exists():
            f_clean = CLEAN_DIR / f"{date_slot}_matched_clean.parquet"
            
        if f_clean.exists():
            df_r = pd.read_parquet(f_raw)
            df_c = pd.read_parquet(f_clean)
            
            r_len, c_len = len(df_r) , len(df_c)
            loss_rate = (r_len - c_len) / r_len * 100
            
            # 格式化时间标签：20181024_d1_0830_0900 -> 0830-0900
            parts = date_slot.split('_')
            time_label = f"{parts[-2]}-{parts[-1]}"
            
            stats.append({'label': time_label, 'rate': loss_rate})
            total_raw += r_len
            total_clean += c_len

    df = pd.DataFrame(stats)
    overall_rate = (total_raw - total_clean) / total_raw * 100
    return df, overall_rate

# ---------------------------------------------------------
# 3. 绘图导出
# ---------------------------------------------------------
def draw_loss_figure(df, overall_rate):
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    
    # 1. 绘制各时段损耗率柱状图
    bars = plt.bar(df['label'], df['rate'], color='#4C72B0', alpha=0.75, 
                   edgecolor='black', linewidth=0.6, label='时段平均损耗率')
    
    # 2. 绘制总损耗率参考线
    plt.axhline(y=overall_rate, color='#C44E52', linestyle='--', linewidth=1.5, 
                label=f'总体平均损耗率 ({overall_rate:.2f}%)')

    # 3. 标注数值 (柱状图上方)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
                 f'{height:.1f}', ha='center', va='bottom', 
                 fontname='Times New Roman', fontsize=9)

    # 4. 优化标签设置
    # plt.title("此处为代码内注释：各时段数据清洗损耗率分布") # 不在图中生成标题
    plt.xlabel('时间区间')
    plt.ylabel('损耗率 (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(prop={'family': 'SimSun', 'size': 9})
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    set_axis_style(ax)
    
    # 5. 导出 SVG
    save_path = SAVE_DIR / "data_loss_analysis_summary.svg"
    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"✅ 汇总图已保存至: {save_path}")

if __name__ == '__main__':
    data_df, global_loss = calculate_metrics()
    if not data_df.empty:
        draw_loss_figure(data_df, global_loss)
    else:
        print("❌ 未找到有效数据文件，请检查路径。")