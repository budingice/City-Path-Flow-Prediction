"""
analyze_path_competition.py
多路径竞争分析脚本

功能:
1. OD 提取：定义路径的物理起终点
2. 竞争组聚类：找出哪些 OD 对下面存在多条高频路径
3. 份额动态计算：计算在 15 分钟片段内，车辆在各条路径间的分配比例

用法示例:
    python analyze_path_competition.py --top-n-od 10 --time-bin 15min
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_kinematics_data(input_dir="path_data"):
    """加载所有路径运动学数据"""
    files = sorted(glob.glob(os.path.join(input_dir, "*_path_kinematics.parquet")))
    if not files:
        raise FileNotFoundError(f"未发现 path_data 目录下的 *_path_kinematics.parquet 文件")

    print(f"📂 正在读取 {len(files)} 个路径运动学文件...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # 确保时间列为 datetime 类型
    if not np.issubdtype(df['start_time'].dtype, np.datetime64):
        df['start_time'] = pd.to_datetime(df['start_time'])

    print(f"✅ 已加载 {len(df)} 条路径记录，时间范围: {df['start_time'].min()} ~ {df['start_time'].max()}")
    return df

def extract_od_pairs(df):
    def safe_parse_sequence(x):
        if isinstance(x, (list, np.ndarray)):
            return x
        if pd.isna(x) or x == "":
            return None
        try:
            # 去除可能存在的空字节或杂质
            clean_x = str(x).replace('\x00', '').strip()
            # 使用 literal_eval 比 eval 安全得多
            return ast.literal_eval(clean_x)
        except Exception:
            return None

    print("🛠️ 正在清洗并解析路径序列...")
    df['path_sequence'] = df['path_sequence'].apply(safe_parse_sequence)
    
    # 删除解析失败的行
    initial_count = len(df)
    df = df.dropna(subset=['path_sequence'])
    df = df[df['path_sequence'].map(len) > 0] # 确保序列不为空
    
    print(f"✅ 清洗完成，剔除了 {initial_count - len(df)} 条损坏记录。")

    # 提取 OD
    df['origin'] = df['path_sequence'].apply(lambda x: x[0])
    df['destination'] = df['path_sequence'].apply(lambda x: x[-1])
    df['od_pair'] = df['origin'].astype(str) + " -> " + df['destination'].astype(str)
    
    return df

def identify_competitive_ods(df):
    """识别存在多路径竞争的 OD 对"""
    print("🏁 正在识别竞争性 OD 对...")

    # 计算每个 OD 对有多少条不同的路径
    od_path_counts = df.groupby('od_pair')['path_signature'].nunique()

    # 筛选出有多条路径的 OD 对
    competitive_ods = od_path_counts[od_path_counts > 1].sort_values(ascending=False)

    print(f"🔍 发现 {len(competitive_ods)} 个存在多路径竞争的 OD 对")

    # 按总流量排序
    od_total_flow = df.groupby('od_pair').size()
    competitive_flow = od_total_flow.loc[competitive_ods.index].sort_values(ascending=False)

    return competitive_ods, competitive_flow


def analyze_multi_path_competition(df, top_n_od=10):
    """
    针对 PNEUMA 数据集的路径竞争定量分析

    :param df: 包含 OD 对的路径数据 DataFrame
    :param top_n_od: 分析的 Top N 竞争 OD 对
    :return: 竞争分析总结 DataFrame 和详细数据
    """
    competitive_ods, competitive_flow = identify_competitive_ods(df)

    # 选取流量最大的 Top N 个竞争组进行深度分析
    top_ods = competitive_flow.head(top_n_od).index

    results = []
    for od in top_ods:
        od_data = df[df['od_pair'] == od].copy()

        # 计算该 OD 下各路径的总流量占比
        total_od_flow = len(od_data)
        path_shares = od_data['path_signature'].value_counts() / total_od_flow

        # 记录结果用于论文表格
        results.append({
            'od_pair': od,
            'total_flow': total_od_flow,
            'num_paths': len(path_shares),
            'main_path_share': path_shares.iloc[0],
            'alt_path_share': path_shares.iloc[1] if len(path_shares) > 1 else 0,
            'entropy': calculate_entropy(path_shares.values)  # 路径选择熵
        })

    return pd.DataFrame(results), df[df['od_pair'].isin(top_ods)]


def calculate_entropy(proportions):
    """计算路径选择的熵（衡量竞争程度）"""
    proportions = np.array(proportions)
    proportions = proportions[proportions > 0]  # 避免 log(0)
    return -np.sum(proportions * np.log(proportions))


def calculate_dynamic_shares(df, time_bin='15min'):
    """
    计算动态路径份额（按时间片段）

    :param df: 详细的竞争 OD 数据
    :param time_bin: 时间聚合粒度
    :return: 动态份额 DataFrame
    """
    print(f"⏰ 正在计算 {time_bin} 粒度的动态路径份额...")

    # 按时间片段聚合
    df['time_bin'] = df['start_time'].dt.floor(time_bin)

    # 按 OD 对和时间片段分组
    dynamic_results = []

    for od_pair in df['od_pair'].unique():
        od_data = df[df['od_pair'] == od_pair].copy()

        # 按时间片段计算路径份额
        for time_slot, group in od_data.groupby('time_bin'):
            total_flow = len(group)
            if total_flow == 0:
                continue

            path_counts = group['path_signature'].value_counts()
            path_shares = path_counts / total_flow

            # 记录主要路径份额
            main_path_share = path_shares.iloc[0] if len(path_shares) > 0 else 0
            alt_path_share = path_shares.iloc[1] if len(path_shares) > 1 else 0

            dynamic_results.append({
                'od_pair': od_pair,
                'time_bin': time_slot,
                'total_flow': total_flow,
                'main_path_share': main_path_share,
                'alt_path_share': alt_path_share,
                'num_paths_used': len(path_shares)
            })

    dynamic_df = pd.DataFrame(dynamic_results)
    print(f"✅ 动态份额计算完成，共 {len(dynamic_df)} 个时间片段")
    return dynamic_df


def plot_competition_summary(competition_summary, output_dir='figures'):
    """可视化竞争分析结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 竞争 OD 对的流量分布
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(competition_summary)), competition_summary['total_flow'])
    plt.xticks(range(len(competition_summary)), competition_summary['od_pair'], rotation=45, ha='right')
    plt.title('Top Competitive OD Pairs by Total Flow')
    plt.ylabel('Total Vehicle Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'competition_od_flow.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 路径份额分布
    plt.figure(figsize=(10, 6))
    plt.scatter(competition_summary['main_path_share'], competition_summary['alt_path_share'],
               s=competition_summary['total_flow']/10, alpha=0.6)
    plt.xlabel('Main Path Share')
    plt.ylabel('Alternative Path Share')
    plt.title('Path Share Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_share_scatter.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_dynamic_shares(dynamic_df, od_pair, output_dir='figures'):
    """可视化特定 OD 对的动态路径份额"""
    os.makedirs(output_dir, exist_ok=True)

    od_data = dynamic_df[dynamic_df['od_pair'] == od_pair].copy()
    if od_data.empty:
        print(f"⚠️ 无数据可用于 OD 对: {od_pair}")
        return

    od_data = od_data.sort_values('time_bin')

    plt.figure(figsize=(14, 6))
    plt.plot(od_data['time_bin'], od_data['main_path_share'], 'b-', label='Main Path Share', linewidth=2)
    plt.plot(od_data['time_bin'], od_data['alt_path_share'], 'r--', label='Alternative Path Share', linewidth=2)
    plt.fill_between(od_data['time_bin'], od_data['main_path_share'], alpha=0.3, color='blue')
    plt.fill_between(od_data['time_bin'], od_data['alt_path_share'], alpha=0.3, color='red')

    plt.title(f'Dynamic Path Shares for OD: {od_pair}')
    plt.xlabel('Time')
    plt.ylabel('Path Share')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = od_pair.replace(' -> ', '_to_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'dynamic_shares_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 70)
    print("🏁 多路径竞争分析器")
    print("=" * 70)

    # 1. 加载数据
    df = load_kinematics_data()

    # 2. OD 提取
    df_with_od = extract_od_pairs(df)

    # 3. 竞争组聚类
    competition_summary, detail_df = analyze_multi_path_competition(df_with_od, top_n_od=10)

    # 4. 份额动态计算
    dynamic_shares = calculate_dynamic_shares(detail_df, time_bin='15min')

    # 5. 输出结果
    print("\n📊 竞争分析总结:")
    print(competition_summary.to_string(index=False))

    # 保存结果
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join('analysis_results', f'path_competition_analysis_{date_str}')
    os.makedirs(output_dir, exist_ok=True)

    competition_summary.to_csv(os.path.join(output_dir, 'competition_summary.csv'), index=False)
    dynamic_shares.to_csv(os.path.join(output_dir, 'dynamic_shares.csv'), index=False)

    print(f"\n💾 结果已保存至 {output_dir}/")

    # 6. 可视化
    print("\n📈 生成可视化图表...")
    plot_competition_summary(competition_summary, output_dir=output_dir)

    # 为前3个 OD 对生成动态份额图
    for od in competition_summary['od_pair'].head(3):
        plot_dynamic_shares(dynamic_shares, od, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("✅ 多路径竞争分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()