import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 设置画图风格
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

def load_experimental_data(results_dir="model_results"):
    """读取训练保存的原始数据"""
    try:
        loss_history = np.load(f"{results_dir}/loss_curve.npy")
        pred_data = np.load(f"{results_dir}/prediction_data.npz")
        y_true = pred_data['y_true']  # [Samples, Horizon, Nodes]
        y_pred = pred_data['y_pred']  # [Samples, Horizon, Nodes]
        max_val = pred_data['max_val']
        return loss_history, y_true, y_pred, max_val
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到结果文件，请确保 step6 已经运行成功。具体错误: {e}")
        return None, None, None, None

def save_refined_metrics(y_true, y_pred, save_dir="model_results"):
    """计算详细的量化指标并保存"""
    horizon = y_true.shape[1]
    with open(f"{save_dir}/metrics_report.txt", "w", encoding='utf-8') as f:
        f.write("📊 STGCN-LSTM 交通流预测性能报告\n")
        f.write("="*40 + "\n")
        
        # 总体指标
        overall_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        overall_rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        f.write(f"总体平均 MAE: {overall_mae:.4f}\n")
        f.write(f"总体平均 RMSE: {overall_rmse:.4f}\n\n")
        
        # 分步指标 (针对多步预测 T+1, T+2, T+3)
        f.write("📈 分步预测分析:\n")
        for h in range(horizon):
            true_h = y_true[:, h, :].flatten()
            pred_h = y_pred[:, h, :].flatten()
            mae = mean_absolute_error(true_h, pred_h)
            rmse = np.sqrt(mean_squared_error(true_h, pred_h))
            r2 = r2_score(true_h, pred_h)
            f.write(f"预测步长 T+{h+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}\n")
            
    print(f"✅ 详细指标报告已生成: {save_dir}/metrics_report.txt")

def plot_training_loss(loss_history):
    """画出模型收敛曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='tab:blue', linewidth=2, label='MSE Loss')
    plt.title("模型训练收敛曲线 (Convergence Curve)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log') # 交通流预测 Loss 通常较小，使用对数坐标更清晰
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("model_results/analysis_loss.png", dpi=300)
    plt.show()

def plot_residual_distribution(y_true, y_pred):
    """残差分布图：分析预测偏置"""
    residuals = (y_true - y_pred).flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='teal', bins=50)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("预测残差分布 (Residual Analysis)")
    plt.xlabel("误差 (实际值 - 预测值)")
    plt.ylabel("频数")
    plt.savefig("model_results/analysis_residuals.png", dpi=300)
    plt.show()

def plot_worst_performing_paths(y_true, y_pred, num_paths=3):
    """【错题本】专门分析误差最大的路径"""
    # 针对第 1 个预测步计算每条路径的平均 MAE
    path_mae = np.mean(np.abs(y_true[:, 0, :] - y_pred[:, 0, :]), axis=0)
    worst_indices = np.argsort(path_mae)[-num_paths:]

    plt.figure(figsize=(16, 5))
    for i, idx in enumerate(reversed(worst_indices)):
        plt.subplot(1, num_paths, i+1)
        # 展示前 100 个样本的变化
        plt.plot(y_true[:100, 0, idx], label='实际', color='gray', alpha=0.6)
        plt.plot(y_pred[:100, 0, idx], label='预测', color='red', linestyle='--')
        plt.title(f"路径 ID: {idx} (最差路径之一)\nMAE: {path_mae[idx]:.2f}")
        plt.legend()
    plt.tight_layout()
    plt.savefig("model_results/analysis_worst_paths.png", dpi=300)
    plt.show()

def plot_random_samples(y_true, y_pred):
    """随机抽取几个样本展示预测拟合度"""
    sample_idx = np.random.randint(0, len(y_true))
    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(50)-0.2, y_true[sample_idx, 0], width=0.4, label='实际流量', color='gray')
    plt.bar(np.arange(50)+0.2, y_pred[sample_idx, 0], width=0.4, label='模型预测', color='blue')
    plt.title(f"随机样本路径流量对比 (样本索引: {sample_idx}, T+1 步)")
    plt.xlabel("路径 ID")
    plt.ylabel("流量值")
    plt.legend()
    plt.savefig("model_results/analysis_sample_bar.png", dpi=300)
    plt.show()

def plot_single_path_time_series(y_true, y_pred, path_idx=0, num_samples=288):
    """
    绘制多时间步单路径对比图 (展示模型随时间演变的能力)
    :param path_idx: 想要观察的特定路段/路径 ID
    :param num_samples: 展示的时间点数量 (例如 288 代表一天内每 5 分钟一个点的数据量)
    """
    # 提取第 path_idx 条路径在第一个预测步 (T+1) 的所有预测值
    # 数据形状: [Samples, Horizon, Nodes] -> 提取为 [Samples]
    true_series = y_true[:num_samples, 0, path_idx]
    pred_series = y_pred[:num_samples, 0, path_idx]
    
    plt.figure(figsize=(16, 6))
    plt.plot(true_series, label='实际流量 (Ground Truth)', color='#34495e', linewidth=2, alpha=0.8)
    plt.plot(pred_series, label='预测流量 (STGCN-LSTM)', color='#e74c3c', linestyle='--', linewidth=1.5)
    
    plt.title(f"路径 ID: {path_idx} 连续时间序列预测对比 (T+1 步长)", fontsize=14)
    plt.xlabel("时间步 (Time Steps)", fontsize=12)
    plt.ylabel("流量值", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 自动保存
    save_name = f"model_results/analysis_time_series_node_{path_idx}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 已生成路径 {path_idx} 的时间序列对比图。")

# --- 修改后的主程序入口 ---
if __name__ == "__main__":
    # 1. 加载数据
    loss, yt, yp, mx = load_experimental_data()
    
    if loss is not None:
        # 2. 生成量化报告
        save_refined_metrics(yt, yp)
        
        # 3. 各种可视化分析
        plot_training_loss(loss)
        plot_residual_distribution(yt, yp)
        plot_worst_performing_paths(yt, yp)
        plot_random_samples(yt, yp)
        
        # --- 新增：调用多时间步单路径对比 ---
        # 你可以指定查看某个特定路径（例如 ID 为 10 的路段）
        # num_samples 决定了横轴展示多长的时间跨度
        plot_single_path_time_series(yt, yp, path_idx=10, num_samples=400)
        
        print("\n✨ 实验分析完成！所有高清图表已保存至 model_results/ 文件夹。")