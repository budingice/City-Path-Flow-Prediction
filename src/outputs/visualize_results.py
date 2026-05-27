import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置科研绘图风格
plt.style.use('seaborn-v0_8-paper') 
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
os.chdir(root_dir) 
print(f"📍 当前工作目录已切换至: {os.getcwd()}")

def load_results(exp_dir, model_tag):
    """加载存放在 npz 中的测试结果"""
    path = os.path.join(exp_dir, f"metrics_{model_tag}_results.npz")
    if not os.path.exists(path):
        print(f"⚠️ 未找到结果文件: {path}")
        return None
    return np.load(path)

def plot_comparison(path_idx=15, time_range=(0, 100)):
    """
    对比不同模型在同一条路径上的预测表现
    :param path_idx: 想要查看的路径索引 (0-49)
    :param time_range: 查看的时间步区间
    """
    # 1. 定义实验路径
    results_map = {
        'Adaptive (mix)': ('experiments/adaptive_run', 'adaptive'),
        'Vanilla (Semantic)': ('experiments/vanilla_semantic_run', 'vanilla_semantic'),
        'Jaccad': ('experiments/vanilla_topo_run', 'vanilla_topo')
    }

    plt.figure(figsize=(12, 6))
    
    # 2. 先画出真实值 (只需要从任意一组中提取)
    first_res = load_results(*list(results_map.values())[0])
    if first_res is None: return
    
    # true shape: (Samples, Horizon, Nodes) -> 我们取第0时刻预测的该节点值
    ground_truth = first_res['true'][:, 0, path_idx]
    plt.plot(ground_truth[time_range[0]:time_range[1]], 
             label='Ground Truth', color='black', linewidth=2, linestyle='--')

    # 3. 循环画出各个模型的预测值
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # 红、蓝、绿
    for i, (label, (d, tag)) in enumerate(results_map.items()):
        res = load_results(d, tag)
        if res is not None:
            pred = res['pred'][:, 0, path_idx]
            plt.plot(pred[time_range[0]:time_range[1]], 
                     label=label, color=colors[i], alpha=0.8)

    plt.title(f'Traffic Flow Prediction Comparison on Path #{path_idx}', pad=15)
    plt.xlabel('Time Step (Minutes)')
    plt.ylabel('Flow (Normalized)')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存图片
    output_path = 'experiments/plots/prediction_comparison.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"📊 对比图已保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    # 运行绘图，查看第 10 条路径在前 120 分钟的表现
    plot_comparison(path_idx=10, time_range=(0, 120))