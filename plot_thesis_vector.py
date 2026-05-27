import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# --- 全局学术绘图参数配置 (Times New Roman & 高清) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.dpi": 600
})

def load_exp_data(exp_root, exp_folders):
    """
    自动从指定文件夹加载预测结果和指标文件[cite: 5]
    """
    results = {}
    for folder in exp_folders:
        path = os.path.join(exp_root, folder)
        if not os.path.exists(path):
            print(f"⚠️ 跳过缺失目录: {folder}")
            continue
            
        # 自动获取该目录下的 .npz 结果文件 (包含 true 和 pred)
        npz_files = [f for f in os.listdir(path) if f.endswith('.npz')]
        if not npz_files:
            continue
            
        data = np.load(os.path.join(path, npz_files[0]))
        
        # 解析 metrics.txt 获取 MAE, RMSE 等数值[cite: 5]
        metrics = {}
        metrics_file = os.path.join(path, "metrics.txt")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding='utf-8') as f:
                for line in f:
                    if ": " in line:
                        k, v = line.strip().split(": ")
                        metrics[k] = float(v)
        
        results[folder] = {
            "true": data['true'], 
            "pred": data['pred'],
            "metrics": metrics
        }
    return results

def plot_comparison_vector(results, path_idx=10, save_dir="experiments/Thesis_Plots"):
    """
    绘制消融实验对比图：展示不同损失函数对波动捕捉的影响[cite: 5]
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 7))
    
    # --- 映射图片中的文件夹名称到论文标签 ---
    # 根据你的图片内容，我们将 base 和 trend 映射为易读的标签
    styles = {
        "adaptive_semantic_base_run": {
            "color": "#d62728", "label": "Baseline (Huber Loss)", "ls": "--", "marker": "s"
        },
        "adaptive_semantic_trend_run": {
            "color": "#1f77b4", "label": "Ours (Trend-Enhanced)", "ls": "-", "marker": "o"
        }
    }

    # 获取真实值作为背景参照
    first_model = list(results.keys())[0]
    true_data = results[first_model]["true"][:, 0, path_idx]
    time_axis = np.arange(len(true_data))
    
    plt.plot(time_axis, true_data, color='black', label='Ground Truth', alpha=0.5, linewidth=1.2)

    # 绘制对比曲线
    for exp_name, style in styles.items():
        if exp_name in results:
            data = results[exp_name]
            pred = data["pred"][:, 0, path_idx]
            plt.plot(time_axis, pred, 
                     color=style["color"], label=style["label"], 
                     linestyle=style["ls"], marker=style["marker"], 
                     markersize=4, markevery=5, linewidth=2)

    plt.title(f"Ablation Study: Prediction Sensitivity on Path {path_idx}", fontsize=16, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Traffic Flow Value", fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 保存矢量图 (SVG) 用于论文插入，PNG 用于快速查看
    for ext in ['svg', 'png']:
        save_path = os.path.join(save_dir, f"ablation_comparison_path_{path_idx}.{ext}")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"🎨 {ext.upper()} 已保存: {save_path}")
    plt.close()

def generate_summary_table(results, save_dir="experiments/Thesis_Plots"):
    """
    导出指标汇总 Excel，方便直接在论文中制作三线表[cite: 5]
    """
    os.makedirs(save_dir, exist_ok=True)
    table_data = []
    for name, data in results.items():
        row = {"Folder": name}
        row.update(data["metrics"])
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    excel_path = os.path.join(save_dir, "ablation_metrics_summary.xlsx")
    df.to_excel(excel_path, index=False)
    
    print("\n" + "="*40)
    print("📊 实验指标对比结果：")
    print(df.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_idx", type=int, default=10, help="选择要可视化的路径编号")
    parser.add_argument("--root", type=str, default="experiments", help="实验根目录")
    args = parser.parse_args()

    # --- 这里必须与你图片中的文件夹名称完全一致 ---
    TARGET_FOLDERS = [
        "adaptive_semantic_base_run", 
        "adaptive_semantic_trend_run"
    ]
    
    data_dict = load_exp_data(args.root, TARGET_FOLDERS)
    
    if data_dict:
        plot_comparison_vector(data_dict, path_idx=args.path_idx)
        generate_summary_table(data_dict)
    else:
        print("❌ 错误：未找到指定文件夹，请检查 --root 路径是否正确。")