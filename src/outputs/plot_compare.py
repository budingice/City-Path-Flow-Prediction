import os
import numpy as np
import matplotlib.pyplot as plt

def plot_merged_comparison(path_idx=10, time_range=(0, 60)):
    # 1. 设置路径（对应你整合后的实验输出文件夹）
    results_dir = "experiments"
    
    # 定义要对比的实验
    # 格式：标签: (文件夹名, 模型前缀)
    compare_list = {
        'Static STGCN (Baseline)': ('vanilla_semantic_run', 'vanilla_semantic'),
        'Adaptive STGCN (Ours)': ('adaptive_run', 'adaptive')
    }

    plt.figure(figsize=(15, 7), dpi=300)
    
    # 2. 加载真实值 (从任意一个结果里取即可)
    sample_path = os.path.join(results_dir, "adaptive_run", "metrics_adaptive_results.npz")
    if not os.path.exists(sample_path):
        print(f"❌ 找不到结果文件，请先运行 train_entry.py。路径尝试: {sample_path}")
        return
        
    data = np.load(sample_path)
    # y_true shape: (Samples, Horizon, Nodes) -> 取第一个预测步，指定路径
    ground_truth = data['true'][:, 0, path_idx]
    
    # 画黑色实线 (真实值)
    plt.plot(ground_truth[time_range[0]:time_range[1]], 
             label='Ground Truth', color='#2c3e50', linewidth=2, zorder=1)

    # 3. 循环画出各个模型的预测线
    styles = {
        'Static STGCN (Baseline)': {'color': '#e74c3c', 'linestyle': '--', 'linewidth': 1.5},
        'Adaptive STGCN (Ours)': {'color': '#3498db', 'linestyle': '-', 'linewidth': 2}
    }

    for label, (folder, prefix) in compare_list.items():
        res_path = os.path.join(results_dir, folder, f"metrics_{prefix}_results.npz")
        if os.path.exists(res_path):
            res_data = np.load(res_path)
            # 取出预测值
            preds = res_data['pred'][:, 0, path_idx]
            plt.plot(preds[time_range[0]:time_range[1]], 
                     label=label, **styles[label], zorder=2)

    # 4. 美化图表
    plt.title(f"Prediction Comparison on Path #{path_idx} (Time Samples 0-60)", fontsize=16, pad=20)
    plt.xlabel("Time Samples", fontsize=14)
    plt.ylabel("Traffic Flow", fontsize=14)
    plt.legend(loc='upper right', frameon=True, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存
    save_path = "model_results/final_comparison_plot.png"
    os.makedirs("model_results", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 新的对比图已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_merged_comparison(path_idx=10, time_range=(0, 60))