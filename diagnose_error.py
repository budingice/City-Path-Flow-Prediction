import torch
import numpy as np

def diagnose_errors_ranking(model, dataset):
    model.eval()
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for x_eval, y_eval in eval_loader:
            pred_eval = model(x_eval.to(current_device))
            y_pred = pred_eval.cpu().numpy() * dataset.max_val
            y_true = y_eval.cpu().numpy() * dataset.max_val

    # 1. 计算每条路径（共50条）的 MAE
    path_errors = np.mean(np.abs(y_true - y_pred), axis=0) # shape: (50,)
    
    # 2. 获取排序索引（从大到小）
    sorted_indices = np.argsort(path_errors)[::-1]

    print("\n" + "="*45)
    print(f"{'Path ID':<10} | {'MAE (误差)':<12} | {'Avg Flow (均值)':<12}")
    print("-" * 45)
    
    for idx in sorted_indices[:10]:  # 查看误差最大的前10条
        avg_flow = np.mean(y_true[:, idx])
        print(f"Path {idx:<5} | {path_errors[idx]:<12.4f} | {avg_flow:<12.4f}")
    print("="*45)

    return sorted_indices, y_true, y_pred

# 调用函数
if __name__ == "__main__":
    sorted_ids, y_true, y_pred = diagnose_errors_ranking(model, dataset)