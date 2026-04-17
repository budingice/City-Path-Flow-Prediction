import os
import glob
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

def list_parquet_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')) + glob.glob(os.path.join(input_dir, '*.parq')))
    return files


def extract_time_label(fname):
    base = os.path.basename(fname)
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[2]
    return os.path.splitext(base)[0]


def batch_traffic_analysis(input_dir='matched_data', output_dir='eda_results', top_k=15):
    os.makedirs(output_dir, exist_ok=True)
    files = list_parquet_files(input_dir)

    all_data = []
    print(f"正在分析 {len(files)} 个快照文件...")

    for f in files:
        df = pd.read_parquet(f)
        if 'edge_id' not in df.columns or 'track_id' not in df.columns:
            raise KeyError(f"文件 {f} 缺少必需列 'edge_id' 或 'track_id'")

        snap_flow = df.groupby('edge_id')['track_id'].nunique().reset_index()
        snap_flow = snap_flow.rename(columns={'track_id': 'flow'})
        snap_flow['time_label'] = extract_time_label(f)
        all_data.append(snap_flow)

    if not all_data:
        raise RuntimeError('No snapshots loaded')

    full_df = pd.concat(all_data, ignore_index=True)

    pivot = full_df.pivot_table(index='edge_id', columns='time_label', values='flow', fill_value=0)

    # CV
    means = pivot.mean(axis=1)
    stds = pivot.std(axis=1)
    cv = stds / (means + 1e-6)

    # Save pivot and cv
    pivot_path = os.path.join(output_dir, 'edge_flow_pivot.csv')
    cv_path = os.path.join(output_dir, 'edge_flow_cv.csv')
    pivot.to_csv(pivot_path)
    cv.to_csv(cv_path, header=['cv'])

    # 1. CV distribution plot
    plt.figure(figsize=(10, 5))
    sns.histplot(cv, bins=50, color='skyblue', kde=True)
    plt.title('路网流量变异系数 (CV) 分布')
    plt.xlabel('CV 值 (波动程度)')
    out_cv_png = os.path.join(output_dir, 'cv_distribution.png')
    plt.savefig(out_cv_png, bbox_inches='tight')
    plt.close()

    # 2. Spatial correlation heatmap for top K mean-flow edges
    top_edges = means.sort_values(ascending=False).head(top_k).index
    corr = pivot.loc[top_edges].T.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title('高流量路径空间相关性矩阵')
    out_corr_png = os.path.join(output_dir, 'spatial_correlation.png')
    plt.savefig(out_corr_png, bbox_inches='tight')
    plt.close()

    # save correlation matrix
    corr_path = os.path.join(output_dir, 'spatial_correlation_matrix.csv')
    corr.to_csv(corr_path)

    print('Saved:')
    print('-', pivot_path)
    print('-', cv_path)
    print('-', out_cv_png)
    print('-', out_corr_png)
    print('-', corr_path)

    return pivot, cv


def main():
    parser = argparse.ArgumentParser(description='Batch EDA for matched_data parquet snapshots')
    parser.add_argument('--input-dir', '-i', default='matched_data', help='Directory containing parquet snapshot files')
    parser.add_argument('--output-dir', '-o', default='eda_results', help='Directory to save EDA outputs')
    parser.add_argument('--top-k', type=int, default=15, help='Top K edges to compute spatial correlation')
    args = parser.parse_args()

    pivot, cv = batch_traffic_analysis(args.input_dir, args.output_dir, top_k=args.top_k)


if __name__ == '__main__':
    main()
