import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    """计算学术标准的评价指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # 为避免分母为0，加一个小偏移量
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1.0))) * 100
    return mae, rmse, mape

def run_comparative_analysis():
    # 1. 读取保存的预测结果 (假设你保存了两个版本)
    try:
        df_lstm = pd.read_csv("model_results/detailed_predictions_stgcn.csv")
    except FileNotFoundError:
        print("❌ 错误：未找到 CSV 结果文件，请先运行训练脚本并保存结果。")
        return

    # 2. 计算各模型的全局指标
    metrics = {}
    models = { 'STGCN-LSTM': df_lstm}
    
    summary_data = []
    for name, df in models.items():
        mae, rmse, mape = calculate_metrics(df['True_Flow'], df['Pred_Flow'])
        metrics[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        summary_data.append([name, mae, rmse, mape])
        print(f"📊 {name} 指标: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

    # 3. 绘制对比柱状图
    res_df = pd.DataFrame(summary_data, columns=['Model', 'MAE', 'RMSE', 'MAPE'])
    res_df.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 6), color=['#3498db', '#e74c3c'])
    plt.title("Model Performance Comparison (MAE & RMSE)")
    plt.ylabel("Vehicle Count")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("model_results/metrics_comparison.png")
    plt.show()

    # 4. 误差分布可视化 (Residual Analysis) 
    plt.figure(figsize=(10, 6))
    for name, df in models.items():
        residuals = df['True_Flow'] - df['Pred_Flow']
        sns.kdeplot(residuals, label=f"{name} Residuals", fill=True)
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Error Residual Distribution (Ideally Centered at 0)")
    plt.xlabel("Prediction Error (Real - Pred)")
    plt.legend()
    plt.show()

# 执行分析
if __name__ == "__main__":
    run_comparative_analysis()