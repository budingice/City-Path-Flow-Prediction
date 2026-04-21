import os
import argparse
from glob import glob
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def list_data_files(data_dir):
    patterns = ["*.parquet", "*.parq", "*.csv"]
    files = []
    for p in patterns:
        files.extend(sorted(glob(os.path.join(data_dir, p))))
    return files


def read_snapshot(fpath):
    ext = os.path.splitext(fpath)[1].lower()
    if ext in ('.parquet', '.parq'):
        df = pd.read_parquet(fpath)
    elif ext == '.csv':
        df = pd.read_csv(fpath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df


def infer_time_label(fname, idx):
    base = os.path.basename(fname)
    # try to extract common pattern tokens separated by '_'
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[2]
    # fallback to filename without extension
    return os.path.splitext(base)[0] + f"_{idx}"


def extract_path_volatility(files, path_col_candidates=('edge_id', 'path_id')):
    all_snaps = []
    print(f"正在处理 {len(files)} 个快照文件，基于 edge_id 统计唯一车辆数...")
    
    for i, f in enumerate(files):
        df = read_snapshot(f)
        # 识别路径列（ edge_id）
        path_col = None
        for c in path_col_candidates:
            if c in df.columns:
                path_col = c
                break
        if path_col is None:
            raise KeyError(f"未找到路径标识列。文件列名为: {df.columns.tolist()}")

        # 针对点位数据，统计该时间片内该路段出现的唯一车辆数 (track_id)
        flow_snap = df.groupby(path_col)['track_id'].nunique().reset_index(name='flow')
        flow_snap = flow_snap.rename(columns={path_col: 'path_id'})
        flow_snap['time_label'] = infer_time_label(f, i)
        all_snaps.append(flow_snap)

    if not all_snaps:
        raise RuntimeError('未能从文件中加载数据')

    full_df = pd.concat(all_snaps, ignore_index=True)
    
    # 透视表：行是路段(path_id)，列是时间(time_label)，缺失值填充为 0
    pivot = full_df.pivot_table(index='path_id', columns='time_label', values='flow', fill_value=0)
    
    # 统计计算
    means = pivot.mean(axis=1)
    stds = pivot.std(axis=1)
    cv = stds / (means + 1e-6)

    stats = pd.DataFrame({
        'path_id': pivot.index,
        'mean_flow': means.values,
        'std_flow': stds.values,
        'cv': cv.values
    }).set_index('path_id')
    # 提取高/低波动 Top 5
    high_volatility = stats.sort_values('cv', ascending=False).head(5)
    low_volatility = stats.sort_values('cv', ascending=True).head(5)

    return stats, high_volatility, low_volatility, pivot

def save_and_report(out_dir, stats, high, low, pivot):
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, 'path_volatility_stats.csv')
    high_path = os.path.join(out_dir, 'high_volatility_top5.csv')
    low_path = os.path.join(out_dir, 'low_volatility_top5.csv')
    pivot_path = os.path.join(out_dir, 'path_flow_timeseries.csv')

    stats.to_csv(stats_path)
    high.to_csv(high_path)
    low.to_csv(low_path)
    pivot.to_csv(pivot_path)

    print(f"Saved stats -> {stats_path}")
    print("\n--- High volatility Top 5 ---")
    print(high)
    print("\n--- Low volatility Top 5 ---")
    print(low)


def plot_examples(out_dir, pivot, high, low):
    if not HAS_MPL:
        print('matplotlib not available; skipping plots')
        return
    os.makedirs(out_dir, exist_ok=True)
    # plot top5
    for name, group in (('high', high), ('low', low)):
        plt.figure(figsize=(10, 6))
        for pid in group.index:
            series = pivot.loc[pid]
            plt.plot(series.index.astype(str), series.values, marker='o', label=str(pid))
        plt.xticks(rotation=45)
        plt.xlabel('time')
        plt.ylabel('flow (count)')
        plt.title(f'{name.capitalize()} volatility paths')
        plt.legend()
        plt.tight_layout()
        out = os.path.join(out_dir, f'{name}_volatility_top5.png')
        plt.savefig(out)
        plt.close()
        print(f'Saved plot {out}')

def verify_integrity(raw_df, path_df):
    raw_vehs = raw_df['track_id'].nunique()
    path_vehs = path_df['flow'].sum() # 这里的逻辑需根据你的聚合方式调整
    
    print(f"原始车辆数: {raw_vehs}")
    # 注意：由于跨快照行驶，path_vehs 可能会略大于 raw_vehs，但数量级必须一致
    if abs(raw_vehs - path_vehs) / raw_vehs > 0.2:
        print("警告：流量数据与原始数据存在显著偏差，请检查清洗阈值！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='matched_data', help='Directory with snapshot files')
    parser.add_argument('--out-dir', default='analysis_results', help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true', help='Do not produce plots')
    args = parser.parse_args()

    files = list_data_files(args.data_dir)
    if not files:
        print('No data files found in', args.data_dir)
        return

    stats, high, low, pivot = extract_path_volatility(files)
    save_and_report(args.out_dir, stats, high, low, pivot)
    if not args.no_plot:
        plot_examples(os.path.join(args.out_dir, 'plots'), pivot, high, low)


if __name__ == '__main__':
    main()
