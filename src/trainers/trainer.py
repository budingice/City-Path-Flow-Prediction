"""
trainer.py - 训练引擎 (通用版)
"""
import torch
import numpy as np
import os

def train_model(model, train_loader, config):
    """
    统一训练接口
    :param model: 待训练模型
    :param train_loader: 数据加载器
    :param config: 配置字典 (包含 device, lr, epochs 等)
    """
    device = config['device']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()
    model.to(device)
    
    losses = []
    print(f"开始训练，设备: {device}...")
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            
            # --- 维度对齐保护 (核心修复) ---
            # 确保预测值和标签的维度一致，防止 MSE 报错或计算错误
            if pred.shape != y.shape:
                # 假设 pred 是 (B, Horizon_3, N), y 是 (B, Horizon_1, N)
                # 取预测的前几步
                loss = criterion(pred[:, :y.shape[1], :], y)
            else:
                loss = criterion(pred, y)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}] | Avg Loss: {avg_loss:.6f}")
            
    return losses

def evaluate_and_save(model, dataset, model_name, save_dir):
    """
    统一评估接口
    """
    model.eval()
    device = next(model.parameters()).device
    # 使用全部数据进行一次性推理
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # 兼容性检查：获取归一化参数
    max_val = getattr(dataset, 'max_val', 1.0)
    
    with torch.no_grad():
        for x, y in loader:
            x_device = x.to(device)
            pred_raw = model(x_device)
            
            # 还原归一化并对齐维度
            pred = pred_raw[:, :y.shape[1], :].cpu().numpy() * max_val
            true = y.numpy() * max_val
            
    # 计算学术指标
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred)**2))
    
    # 保存原始数据用于 evaluator.py 绘图
    os.makedirs(save_dir, exist_ok=True)
    np.savez(f"{save_dir}/{model_name}_results.npz", true=true, pred=pred)
    
    # 保存文本结果
    with open(f"{save_dir}/{model_name}_metrics.txt", "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}")
    
    print(f"✅ {model_name} 实验完成。指标: MAE={mae:.4f}, RMSE={rmse:.4f}")