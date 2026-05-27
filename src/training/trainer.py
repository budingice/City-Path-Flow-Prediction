import torch
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
from .losses import MultiStatLoss

class Trainer:
    def __init__(self, model, config, adj_matrix=None, loss_mode='base'):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 兼容性处理：只有 Torch 模型才移动到 GPU
        if isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
            self.is_torch_model = True
        else:
            self.model = model
            self.is_torch_model = False
            
        self.adj = adj_matrix.to(self.device) if adj_matrix is not None else None
        
        # 只有 Torch 模型需要优化器
        if self.is_torch_model:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=config['lr'], 
                weight_decay=config.get('weight_decay', 0)
            )
        
        self.criterion = MultiStatLoss(
            loss_type=loss_mode, 
            alpha=config.get('loss_alpha', 0.1)
        )
        
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'alpha_evolution': []}

    def fit(self, train_loader, val_loader, epochs, save_path):
        if not self.is_torch_model:
            return
            
        print(f"🚀 开始训练 | 设备: {self.device} | 模式: {self.criterion.loss_type}")
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for x, y, mask in train_loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                self.optimizer.zero_grad()
                
                try:
                    pred = self.model(x, self.adj)
                except TypeError:
                    pred = self.model(x)
                
                loss = self.criterion(pred, y, mask)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            val_loss = self.evaluate_loss(val_loader)
            avg_train_loss = np.mean(train_losses)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 记录自适应权重的演化
            if hasattr(self.model, 'alpha'):
                self.history['alpha_evolution'].append(torch.sigmoid(self.model.alpha).item())

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
        
        with open(save_path.replace('.pth', '_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

    def evaluate_loss(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y, mask in loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                try:
                    pred = self.model(x, self.adj)
                except TypeError:
                    pred = self.model(x)
                loss = self.criterion(pred, y, mask)
                losses.append(loss.item())
        return np.mean(losses)

    def test(self, test_loader, max_val, save_dir, model_name="model"):
        if self.is_torch_model:
            model_path = os.path.join(save_dir, "best_model.pth")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            
        all_preds, all_trues, all_masks = [], [], []
        with torch.no_grad():
            for x, y, mask in test_loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                
                # 兼容 HA_Baseline 的 predict 方法
                if hasattr(self.model, 'predict'):
                    pred = self.model.predict(x)
                else:
                    try:
                        pred = self.model(x, self.adj)
                    except TypeError:
                        pred = self.model(x)
                
                all_preds.append(pred.cpu().numpy() * max_val)
                all_trues.append(y.cpu().numpy() * max_val)
                all_masks.append(mask.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        valid_idx = (masks > 0)
        if np.sum(valid_idx) == 0:
            return 0, 0, 0

        mae = np.mean(np.abs(preds[valid_idx] - trues[valid_idx]))
        rmse = np.sqrt(np.mean((preds[valid_idx] - trues[valid_idx])**2))
        wape = np.sum(np.abs(preds[valid_idx] - trues[valid_idx])) / (np.sum(np.abs(trues[valid_idx])) + 1e-5)

        print(f"📊 {model_name} 测试结果: MAE: {mae:.4f}, RMSE: {rmse:.4f}, WAPE: {wape:.4f}")
        results_dir = os.path.join("experiments", "predictions")
        os.makedirs(results_dir, exist_ok=True)

        # 将预测值和真实值保存为 npz 格式，方便后续绘图
        save_fn = os.path.join(results_dir, f"{model_name}_results.npz")
        np.savez(save_fn, preds=preds, trues=trues, masks=masks)
        print(f"💾 详细预测结果已保存至: {save_fn}")
                
        # 结果汇总到 CSV
        summary_path = "experiments/benchmark_summary.csv"
        os.makedirs("experiments", exist_ok=True)
        res_df = pd.DataFrame([{"Model": model_name, "MAE": mae, "RMSE": rmse, "WAPE": wape}])
        if os.path.exists(summary_path):
            try:
                old_df = pd.read_csv(summary_path)
                res_df = pd.concat([old_df, res_df], ignore_index=True).drop_duplicates(subset=['Model'], keep='last')
            except: pass
        res_df.to_csv(summary_path, index=False)
        return mae, rmse, wape