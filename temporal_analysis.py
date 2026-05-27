import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

def load_config(config_path="configs/config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join("..", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calculate_wape(y_true, y_pred):
    """计算 WAPE 指标 (Weighted Absolute Percentage Error)"""
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-6)

def run_temporal_analysis():
    # 1. 配置加载
    config = load_config()
    exp_root = config['path']['exp_root']
    plot_root = config['path'].get('plot_root', 'experiments/Thesis_Plots')
    os.makedirs(plot_root, exist_ok=True)

    model_paths = {
        'Adaptive': os.path.join(exp_root, "adaptive_semantic_run", "adaptive_semantic_results.npz"),
        '拓扑 (Topo)': os.path.join(exp_root, "vanilla_topo_run", "vanilla_topo_results.npz"),
        '语义 (Semantic)': os.path.join(exp_root, "vanilla_semantic_run", "vanilla_semantic_results.npz")
    }

    # 2. 核心逻辑：按时间步切分窗口
    # 假设你的 y_true 维度是 [Samples, Paths] 或 [Samples, Horizon, Paths]
    # 我们需要根据 Sample 的索引对 30 分钟周期取模
    # pNEUMA: 1 step = 1 min, period = 30
    period = 30
    
    temporal_results = []

    for label, path in model_paths.items():
        if not os.path.exists(path): continue
        data = np.load(path)
        y_true = data['true']
        y_pred = data['pred']
        mask = data['mask']

        # 统一维度为 [Total_Steps, Paths]
        if y_true.ndim == 3: # [Samples, Horizon, Paths]
            y_true = y_true[:, 0, :] # 取预测的第一步作为代表
            y_pred = y_pred[:, 0, :]
            mask = mask[:, 0, :]

        # 遍历每一个样本，判断它处于 30min 周期内的哪个位置
        # 注意：这里假设 Samples 是按时间顺序排列的连续序列
        num_samples = y_true.shape[0]
        time_indices = np.arange(num_samples) % period

        # 定义三个阶段 (有效采样是 0-14 分钟)
        phases = {
            '1. Start (0-5m)': (time_indices >= 0) & (time_indices < 5),
            '2. Mid (5-10m)': (time_indices >= 5) & (time_indices < 10),
            '3. End (10-15m)': (time_indices >= 10) & (time_indices < 15)
        }

        for phase_name, condition in phases.items():
            # 仅在 mask == 1 且符合时间段的情况下计算
            valid_mask = (mask > 0) & condition[:, np.newaxis]
            if np.sum(valid_mask) > 0:
                wape = calculate_wape(y_true[valid_mask], y_pred[valid_mask])
                temporal_results.append({
                    'Model': label,
                    'Phase': phase_name,
                    'WAPE': wape
                })

    # 3. 生成报表
    df_plot = pd.DataFrame(temporal_results)
    report = df_plot.pivot(index='Phase', columns='Model', values='WAPE')

    # 4. 导出 TXT 表格
    txt_path = os.path.join(plot_root, "temporal_wape_table.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Table: Temporal WAPE Distribution (Cold Start vs Stable Stages)\n")
        f.write("-" * 80 + "\n")
        f.write(report.to_string())
        f.write("\n" + "-" * 80 + "\n")
        f.write("Note: Start Phase represents 'Cold Start' after 15min data missing.\n")

    print(f"✅ 时间维度误差表已导出: {txt_path}")

    # 5. 绘图：折线对比图
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_plot, x='Phase', y='WAPE', hue='Model', marker='o', linewidth=2.5)
    plt.title('采样周期内的误差波动 (WAPE)', fontsize=14)
    plt.ylabel('WAPE (越低越好)', fontsize=12)
    plt.xlabel('采样窗口阶段', fontsize=12)
    
    fig_path = os.path.join(plot_root, "temporal_error_fluctuation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ 时间波动分析图已保存: {fig_path}")

if __name__ == "__main__":
    run_temporal_analysis()