from basicsr.utils.registry import LOSS_REGISTRY
import torch
import torch.nn.functional as F
from torch import nn


def soft_histogram(x, bins, min_val, max_val, sigma):
    delta = float(max_val - min_val) / float(bins)
    centers = float(min_val) + delta * (torch.arange(bins).float() + 0.5)
    centers = centers.to(x.device)
    x = torch.unsqueeze(x, 0) - torch.unsqueeze(centers, 1)
    x = torch.sigmoid(sigma * (x + delta / 2)) - torch.sigmoid(sigma * (x - delta / 2))
    histogram = x.sum(dim=1)
    return histogram

def compute_cdf(histogram):
    # 累加直方图以计算CDF
    cdf = torch.cumsum(histogram, dim=0)
    # 归一化
    cdf = cdf / cdf[-1]
    return cdf

@LOSS_REGISTRY.register()
class CDFL1HistogramLoss(nn.Module):
    def __init__(self, bins=256, sigma=300, loss_weight=1.0, size=256):
        super(CDFL1HistogramLoss, self).__init__()
        self.bins = bins
        self.sigma = sigma
        self.size = size
        self.loss_weight = loss_weight
        
    def forward(self, pred, target, weight=None, **kwargs):
        pred = F.interpolate(pred, size=self.size, mode='bilinear', align_corners=False)
        target = F.interpolate(target, size=self.size, mode='bilinear', align_corners=False)
      
        N, C, H, W = pred.shape
        pred = pred.view(N, C, -1)
        target = target.view(N, C, -1)

        loss = []

        for n in range(N):
            img1 = pred[n]
            img2 = target[n]
            for i in range(C):
                # 计算直方图
                img1_hist = soft_histogram(img1[i], self.bins, min_val=0.0, max_val=1.0, sigma=self.sigma)
                img2_hist = soft_histogram(img2[i], self.bins, min_val=0.0, max_val=1.0, sigma=self.sigma)
                
                # 计算CDF
                img1_cdf = compute_cdf(img1_hist)
                img2_cdf = compute_cdf(img2_hist)
                
                # 计算CDF之间的L1距离
                loss.append(F.l1_loss(img1_cdf, img2_cdf))
        
        # 计算最终的平均损失
        loss = sum(loss) / (N * C)
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class GrayWorldLoss(nn.Module):
    """Gray World Assumption Loss with size control.
    
    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Reduction mode ('mean' or 'sum'). Default: 'mean'.
        eps (float): Small value to avoid division by zero. Default: 1e-8.
        size (int): Target spatial size for resizing. Default: 256.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-8, size=256):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.size = size
        
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute Gray World Loss with resizing.
        
        Args:
            pred (Tensor): Predicted image tensor of shape (B, C, H, W).
            
        Returns:
            Tensor: Scalar loss value.
        """
        # Resize input to control computation
        # if self.size is not None and (pred.shape[2] != self.size or pred.shape[3] != self.size):
        #     pred = F.interpolate(
        #         pred, 
        #         size=(self.size, self.size), 
        #         mode='bilinear', 
        #         align_corners=False
        #     )
        
        # Calculate channel means (B, C)
        channel_means = torch.mean(pred, dim=(2, 3))  # Shape: [B, C]
        
        # Compute squared differences
        diff_rg = torch.pow(channel_means[:, 0] - channel_means[:, 1], 2)
        diff_gb = torch.pow(channel_means[:, 1] - channel_means[:, 2], 2)
        diff_br = torch.pow(channel_means[:, 2] - channel_means[:, 0], 2)
        
        # L2-norm form
        loss = torch.sqrt(diff_rg + diff_gb + diff_br + self.eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return self.loss_weight * loss

