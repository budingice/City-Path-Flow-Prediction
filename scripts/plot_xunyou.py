import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import re

def setup_academic_style():
    """配置符合论文要求的字体和样式"""
    # 宋体用于中文，Times New Roman 用于数字和英文
    plt.rcParams['font.sans-serif'] = ['SimSun'] 
    plt.rcParams['axes.unicode_minus'] = False 
    matplotlib.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def process_data(csv_path):
    """筛选15x15数据并解析参数"""
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return pd.DataFrame()

    # 1. 读取原始 CSV
    df = pd.read_csv(csv_path, header=None, names=['Model', 'MAE', 'RMSE', 'WAPE'])
    
    # 2. 【核心筛选】只保留本次 15x15 的网格寻优数据，排除旧数据
    df = df[df['Model'].str.contains('GridSearch_15x15', na=False)].copy()
    
    if df.empty:
        print("⚠️ 未在 CSV 中找到包含 'GridSearch_15x15' 的数据！")
        return df

    # 3. 强制数值化指标列
    for col in ['MAE', 'RMSE', 'WAPE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. 正则表达式解析参数
    def extract_params(row):
        if not isinstance(row, str): return pd.Series([np.nan, np.nan])
        h = re.search(r'h(\d+)', row)
        lr = re.search(r'lr([\d\.]+)', row)
        return pd.Series([
            int(h.group(1)) if h else np.nan, 
            float(lr.group(1)) if lr else np.nan
        ])

    df[['h_dim', 'lr']] = df['Model'].apply(extract_params)
    df = df.dropna(subset=['h_dim', 'lr', 'MAE'])
    
    # 找到全局最优参数（MAE最低）
    best_idx = df['MAE'].idxmin()
    print(f"⭐ 本次寻优最优组合: Hidden_Dim={int(df.loc[best_idx, 'h_dim'])}, LR={df.loc[best_idx, 'lr']:.6f}")
    print(f"⭐ 对应最低 MAE: {df.loc[best_idx, 'MAE']:.4f}")
    
    return df

def plot_3d_surface(df, save_dir):
    """绘制 3D 响应曲面图"""
    # 聚合数据并透视
    df_pivot = df.groupby(['lr', 'h_dim'])['MAE'].mean().unstack()
    X_raw = df_pivot.columns.values  # h_dim
    Y_raw = df_pivot.index.values    # lr
    Z = df_pivot.values

    # 对学习率取对数，使3D空间网格分布均匀
    X, Y = np.meshgrid(X_raw, np.log10(Y_raw))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.5, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('\n隐藏层维度 (Hidden Dim)', fontproperties='SimSun')
    ax.set_ylabel('\n学习率 (Log10 LR)', fontproperties='SimSun')
    ax.set_zlabel('\nMAE', fontproperties='Times New Roman')
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1).set_label('MAE')
    ax.view_init(elev=30, azim=135)
    
    plt.savefig(os.path.join(save_dir, '3d_surface_15x15.png'), dpi=600, bbox_inches='tight')
    print("✅ 3D曲面图已生成")

def plot_best_param_lines(df, save_dir):
    """在最优参数下画两个维度的对比图"""
    # 找到全局 MAE 最低时对应的参数
    best_row = df.loc[df['MAE'].idxmin()]
    best_h = best_row['h_dim']
    best_lr = best_row['lr']

    # 1. 固定最优学习率，画维度变化图
    df_h = df[df['lr'] == best_lr].sort_values('h_dim')
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_h['h_dim'], df_h['MAE'], 'b-o', label='MAE', linewidth=1.5)
    ax1.set_xlabel(f'隐藏层维度 (固定最优 LR={best_lr:.6f})', fontproperties='SimSun')
    ax1.set_ylabel('MAE', color='b')
    ax2 = ax1.twinx()
    ax2.plot(df_h['h_dim'], df_h['RMSE'], 'r--s', label='RMSE', linewidth=1.5)
    ax2.set_ylabel('RMSE', color='r')
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'line_h_dim_at_best_lr.png'), dpi=600)

    # 2. 固定最优维度，画学习率变化图
    df_lr = df[df['h_dim'] == best_h].sort_values('lr')
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_lr['lr'], df_lr['MAE'], 'b-o', label='MAE', linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel(f'学习率 (固定最优 Hidden_Dim={int(best_h)})', fontproperties='SimSun')
    ax1.set_ylabel('MAE', color='b')
    ax2 = ax1.twinx()
    ax2.plot(df_lr['lr'], df_lr['RMSE'], 'r--s', label='RMSE', linewidth=1.5)
    ax2.set_ylabel('RMSE', color='r')
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'line_lr_at_best_h.png'), dpi=600)
    print("✅ 两个最优参数切面折线图已生成")

if __name__ == "__main__":
    CSV_FILE = "experiments/benchmark_summary.csv"
    SAVE_DIR = "experiments"
    
    setup_academic_style()
    data = process_data(CSV_FILE)
    
    if not data.empty:
        plot_3d_surface(data, SAVE_DIR)
        plot_best_param_lines(data, SAVE_DIR)