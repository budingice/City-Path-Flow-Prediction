import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保类名是 MultiStatLoss
class MultiStatLoss(nn.Module):
    def __init__(self, loss_type='base', alpha=0.5):
        super(MultiStatLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha 
        # 兼容原有的 HuberLoss 设定
        self.base_loss = nn.HuberLoss(delta=1.0, reduction='none')

    def forward(self, pred, true, mask):
        # 计算基础 Huber 误差并应用掩码[cite: 9]
        huber_err = self.base_loss(pred, true)
        
        if self.loss_type == 'trend':
            # 趋势项计算：沿时间步维度（dim=1）做差分[cite: 7, 8]
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            true_diff = true[:, 1:, :] - true[:, :-1, :]
            trend_err = torch.mean(torch.abs(pred_diff - true_diff))
            return torch.mean(huber_err * mask) + self.alpha * trend_err
            
        elif self.loss_type == 'weighted':
            # 加权项计算[cite: 6]
            weight = 1.0 + true 
            return torch.mean(huber_err * weight * mask)

        return torch.mean(huber_err * mask)