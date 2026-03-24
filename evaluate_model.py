import torch
import numpy as np

def evaluate_without_sklearn(model, dataset, raw_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 获取全部测试数据
    # 假设你的 dataset 已经包含了所有样本
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # 反归一化：将 0-1 回传到原始车辆数
            y_pred = pred.cpu().numpy() * dataset.max_val
            y_true = y.cpu().numpy() * dataset.max_val

    # --- 手动计算核心指标 ---
    
    # 1. MAE (Mean Absolute Error)
    # 公式：sum(|y_true - y_pred|) / n
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 2. RMSE (Root Mean Squared Error)
    # 公式：sqrt(sum((y_true - y_pred)^2) / n)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # 3. MAPE (Mean Absolute Percentage Error)
    # 公式：(100% / n) * sum(|(y_true - y_pred) / y_true|)
    # 注意：为了防止真实值为 0 导致除法溢出，我们在分母加一个极小值或 1.0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1.0))) * 100

    print("\n" + "="*30)
    print("📊 交通流量预测模型 - 误差报告 (Numpy版)")
    print("="*30)
    print(f"📍 MAE  : {mae:.4f} 辆/5min")
    print(f"📍 RMSE : {rmse:.4f}")
    print(f"📍 MAPE : {mape:.2f}%")
    print("="*30)
    
    return mae, rmse, mape

# 在训练脚本最后调用
evaluate_without_sklearn(model, dataset, raw_data)