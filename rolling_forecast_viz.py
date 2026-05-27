import os
import numpy as np
import matplotlib.pyplot as plt

def plot_rolling_forecast(exp_dir, path_idx=10, start_step=0, num_steps=100, save_dir="experiments/Thesis_Plots"):
    # 1. 加载数据
    npz_path = [f for f in os.listdir(exp_dir) if f.endswith('.npz')][0]
    data = np.load(os.path.join(exp_dir, npz_path))
    
    # true/pred 形状: [Samples, Horizon, Nodes]
    true = data['true'][:, :, path_idx]
    pred = data['pred'][:, :, path_idx]
    
    # 2. 准备画布
    plt.figure(figsize=(16, 6), dpi=300)
    
    # 3. 绘制 Ground Truth (作为背景)
    # 连续取出每个时刻的第一个预测目标值作为真实曲线
    ground_truth = true[start_step:start_step+num_steps, 0]
    plt.plot(range(num_steps), ground_truth, color='black', label='Actual Flow', alpha=0.3, linewidth=3)
    
    # 4. 模拟滚动预测过程
    # 我们每隔几个步长画一段预测线，体现“滚一遍”的感觉
    interval = 5  # 每隔5个步长显示一段预测
    horizon = true.shape[1]
    
    for i in range(0, num_steps - horizon, interval):
        # 提取当前时刻发出的预测段 [i, i + horizon]
        current_pred = pred[start_step + i, :]
        
        # 绘制这一小段预测线
        color = '#1f77b4' if i == 0 else plt.cm.Blues(0.4 + 0.6 * (i/num_steps))
        plt.plot(range(i, i + horizon), current_pred, color=color, alpha=0.8, linewidth=1.5)
        
        # 在预测起点打个小点，体现预测的“发射”位置
        plt.scatter(i, current_pred[0], color=color, s=15, alpha=0.8)

    plt.title(f"Rolling Horizon Forecast Visualization (Path {path_idx})", fontsize=14)
    plt.xlabel("Time Samples (Future Progression)", fontsize=11)
    plt.ylabel("Traffic Flow", fontsize=11)
    plt.legend(["Actual Flow", "Model Predictions (Multi-step)"], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "rolling_forecast_viz.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()
    print(f"🎬 滚动预测动态示意图已保存: {out_path}")

if __name__ == "__main__":
    # 使用你效果最好的那个模型实验目录
    BEST_EXP = "experiments/adaptive_semantic_run"
    if os.path.exists(BEST_EXP):
        plot_rolling_forecast(BEST_EXP, path_idx=10, num_steps=120)
    else:
        print("❌ 找不到实验数据目录，请确认路径。")