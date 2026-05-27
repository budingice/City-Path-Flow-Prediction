import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

def load_config(config_path="configs/config.yaml"):
    """从本地读取项目配置文件"""
    if not os.path.exists(config_path):
        # 如果当前目录找不到，向上找一级
        config_path = os.path.join("..", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_results(path):
    if not os.path.exists(path):
        print(f"❌ 找不到结果文件: {path}")
        return None, None, None
    data = np.load(path)
    keys = list(data.files)
    true_key = 'true' if 'true' in keys else keys[0]
    pred_key = 'pred' if 'pred' in keys else keys[1]
    mask_key = 'mask' if 'mask' in keys else (keys[2] if len(keys)>2 else None)
    
    y_true = data[true_key]
    y_pred = data[pred_key]
    mask = data[mask_key] if mask_key is not None else np.ones_like(y_true)
    return y_true, y_pred, mask

def run_thesis_analysis():
    # 1. 配置解析
    config = load_config()
    exp_root = config['path']['exp_root']
    plot_root = config['path'].get('plot_root', os.path.join(exp_root, "Thesis_Plots"))
    cv_thresholds = config['analysis'].get('cv_thresholds', [0.6, 1.2])
    
    # 确保导出目录存在
    os.makedirs(plot_root, exist_ok=True)

    # 2. 模型结果路径映射
    model_paths = {
        'Adaptive': os.path.join(exp_root, "adaptive_semantic_run", "adaptive_semantic_results.npz"),
        '拓扑 (Topo)': os.path.join(exp_root, "vanilla_topo_run", "vanilla_topo_results.npz"),
        '语义 (Semantic)': os.path.join(exp_root, "vanilla_semantic_run", "vanilla_semantic_results.npz")
    }

    y_preds = {}
    y_true_ref = None
    mask_ref = None

    print(f"📂 正在读取实验数据从: {exp_root}")
    for label, p in model_paths.items():
        yt, yp, mk = load_results(p)
        if yt is not None:
            y_preds[label] = yp
            y_true_ref = yt
            mask_ref = mk

    if y_true_ref is None:
        print("🛑 未发现任何有效的 .npz 结果文件，请检查 experiments 文件夹结构。")
        return

    # --- 3. 指标计算 (维度降为 [Paths]) ---
    reduce_axes = tuple(range(y_true_ref.ndim - 1))
    y_true_nan = np.where(mask_ref > 0, y_true_ref, np.nan)
    
    mean_flow = np.nanmean(y_true_nan, axis=reduce_axes).flatten()
    std_flow = np.nanstd(y_true_nan, axis=reduce_axes).flatten()
    cv = std_flow / (mean_flow + 1e-6)

    mae_results = {}
    for label, yp in y_preds.items():
        yp_nan = np.where(mask_ref > 0, yp, np.nan)
        mae_results[label] = np.nanmean(np.abs(y_true_nan - yp_nan), axis=reduce_axes).flatten()

    # --- 4. 构建异质性分组 ---
    df = pd.DataFrame({'Path_ID': np.arange(len(cv)), 'CV': cv})
    for label, mae_val in mae_results.items():
        df[f'MAE_{label}'] = mae_val

    def get_group(x):
        if x < cv_thresholds[0]: return f'1. Stable (CV < {cv_thresholds[0]})'
        if x < cv_thresholds[1]: return f'2. Moderate'
        return f'3. Volatile (CV > {cv_thresholds[1]})'
    df['Group'] = df['CV'].apply(get_group)

    # --- 5. 聚合报表 ---
    report = df.groupby('Group').agg({
        'Path_ID': 'count',
        'CV': 'mean',
        'MAE_拓扑 (Topo)': 'mean',
        'MAE_语义 (Semantic)': 'mean',
        'MAE_Adaptive': 'mean'
    }).rename(columns={'Path_ID': 'Count', 'CV': 'Avg_CV'})

    # --- 6. 导出论文风格 TXT ---
    txt_save_path = os.path.join(plot_root, "heterogeneity_table.txt")
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write("Table: MAE Performance by Path Heterogeneity Groups\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Group':<25} {'Count':<8} {'Avg_CV':<10} {'MAE_Topo':<12} {'MAE_Sem':<12} {'MAE_Ada':<12}\n")
        f.write("-" * 90 + "\n")
        for idx, row in report.iterrows():
            f.write(f"{idx:<25} {int(row['Count']):<8} {row['Avg_CV']:<10.4f} "
                    f"{row['MAE_拓扑 (Topo)']:<12.4f} {row['MAE_语义 (Semantic)']:<12.4f} {row['MAE_Adaptive']:<12.4f}\n")
        f.write("-" * 90 + "\n")

    print(f"✅ 学术数据表已保存: {txt_save_path}")

    # --- 7. 可视化绘图 ---
    plot_df = report[['MAE_拓扑 (Topo)', 'MAE_语义 (Semantic)', 'MAE_Adaptive']].stack().reset_index()
    plot_df.columns = ['Group', 'Model', 'MAE']
    plot_df['Model'] = plot_df['Model'].map({'MAE_拓扑 (Topo)': '拓扑 (Topo)', 
                                           'MAE_语义 (Semantic)': '语义 (Semantic)', 
                                           'MAE_Adaptive': 'Adaptive'})

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x='Group', y='MAE', hue='Model', palette='deep')
    plt.title('不同路径流量波动水平下的模型预测性能对比', fontsize=14, pad=15)
    plt.ylabel('平均 MAE (排除缺失时段)', fontsize=12)
    plt.xlabel('路径分组 (基于变异系数 CV)', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    # 保存图片到 Thesis_Plots
    fig_save_path = os.path.join(plot_root, "heterogeneity_comparison.png")
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 论文插图已保存: {fig_save_path}")
    print("\n" + report.round(4).to_string())

if __name__ == "__main__":
    run_thesis_analysis()