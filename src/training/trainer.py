import torch
import numpy as np
import os
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, adj_matrix=None):
        """
        :param model: STGCN 模型实例
        :param config: 来自 yaml 的 train 配置块
        :param adj_matrix: 训练所需的邻接矩阵 (已移动到 device)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.adj = adj_matrix.to(self.device) if adj_matrix is not None else None
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 0)
        )
        self.criterion = torch.nn.HuberLoss(delta=1.0)
        
        # 用于早停逻辑
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            # STGCN 通常需要 adj 作为输入
            pred = self.model(x, self.adj)
            
            # 维度对齐保护
            if pred.shape != y.shape:
                loss = self.criterion(pred[:, :y.shape[1], :], y)
            else:
                loss = self.criterion(pred, y)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x, self.adj)
                
                if pred.shape != y.shape:
                    loss = self.criterion(pred[:, :y.shape[1], :], y)
                else:
                    loss = self.criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, save_path):
        """完整的训练流程，包含早停"""
        print(f"🚀 开始训练，运行设备: {self.device}")
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch [{epoch+1:03d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # 早停与最佳模型保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.get('early_stop', 15):
                print(f"⏱️ Early stopping triggered at epoch {epoch+1}")
                break

    def test(self, test_loader, max_val, save_dir, model_name="STGCN"):
        """在测试集上评估并产出论文级结果"""
        self.model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
        self.model.eval()
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                pred = self.model(x, self.adj)
                
                # 对齐并反归一化
                pred_rescaled = pred[:, :y.shape[1], :].cpu().numpy() * max_val
                true_rescaled = y.numpy() * max_val
                
                all_preds.append(pred_rescaled)
                all_trues.append(true_rescaled)
        
        pred_final = np.concatenate(all_preds, axis=0)
        true_final = np.concatenate(all_trues, axis=0)
        
        # 计算指标
        mae = np.mean(np.abs(true_final - pred_final))
        rmse = np.sqrt(np.mean((true_final - pred_final)**2))
        mape = np.mean(np.abs((true_final - pred_final) / (true_final + 1e-5))) # 防止除零

        # 保存结果用于后期绘图
        res_path = os.path.join(save_dir, f"{model_name}_results.npz")
        np.savez(res_path, true=true_final, pred=pred_final)
        
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.4f}")
            
        print(f"✅ 实验结束 | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        return mae, rmse