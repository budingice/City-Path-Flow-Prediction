import numpy as np
import matplotlib.pyplot as plt
import os

def plot_ablation_comparison(res_static, res_adaptive, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    data_s = np.load(res_static)
    data_a = np.load(res_adaptive)
    
    true = data_s['true']  # [Samples, Horizon, Nodes]
    pred_s = data_s['pred']
    pred_a = data_a['pred']
    
    # 选取第 0 个样本，预测的第 0 步，第 10 个节点（路径）进行对比展示
    sample_idx = 0
    step_idx = 0
    node_idx = 10 
    
    plt.figure(figsize=(12, 6))
    
    # 绘制真实值
    plt.plot(true[:, step_idx, node_idx], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
    # 绘制静态模型预测值
    plt.plot(pred_s[:, step_idx, node_idx], label='Static STGCN (Baseline)', color='red', linestyle='--', alpha=0.8)
    # 绘制自适应模型预测值
    plt.plot(pred_a[:, step_idx, node_idx], label='Adaptive STGCN (Ours)', color='blue', linestyle='-', alpha=0.8)
    
    plt.title(f"Prediction Comparison on Path {node_idx} (Step {step_idx+1})", fontsize=14)
    plt.xlabel("Time Samples", fontsize=12)
    plt.ylabel("Traffic Flow", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(f"{save_dir}/ablation_comparison_plot.png", dpi=300)
    print(f"📊 对比图表已保存至: {save_dir}/ablation_comparison_plot.png")
    plt.show()

def plot_advanced_analysis(res_static, res_adaptive, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    ds = np.load(res_static)
    da = np.load(res_adaptive)
    
    # 计算所有样本、所有路径的平均绝对误差
    error_s = np.abs(ds['true'] - ds['pred']).mean(axis=1) # [Samples, Nodes]
    error_a = np.abs(da['true'] - da['pred']).mean(axis=1) # [Samples, Nodes]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图 1: 误差改善热力图
    improvement = error_s - error_a
    im = axes[0].imshow(improvement.T, aspect='auto', cmap='RdYlGn')
    axes[0].set_title("Error Improvement (Static - Adaptive)\nGreen means Adaptive is better", fontsize=12)
    axes[0].set_ylabel("Path Index")
    axes[0].set_xlabel("Time Samples")
    fig.colorbar(im, ax=axes[0])
    
    # 子图 2: 残差分布对比
    res_s = (ds['true'] - ds['pred']).flatten()
    res_a = (da['true'] - da['pred']).flatten()
    axes[1].hist(res_s, bins=50, alpha=0.5, label='Static Residuals', color='red', density=True)
    axes[1].hist(res_a, bins=50, alpha=0.5, label='Adaptive Residuals', color='blue', density=True)
    axes[1].set_title("Residual Distribution Comparison", fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/academic_analysis.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_ablation_comparison("model_results/Static_Model_results.npz", 
                             "model_results/Adaptive_Model_results.npz")
    
    plot_advanced_analysis("model_results/Static_Model_results.npz", 
                           "model_results/Adaptive_Model_results.npz", 
                           save_dir="plots")