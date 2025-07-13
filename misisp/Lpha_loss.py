import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.archs.vgg_arch import VGGFeatureExtractor

class LphaLoss(nn.Module):
    def __init__(self, vgg_layer='conv3_1', block_size=32, cosine_threshold=0.2, device='cuda'):
        super().__init__()
        self.block_size = block_size
        self.cosine_threshold = cosine_threshold
        self.device = device
        self.vgg = VGGFeatureExtractor(
            layer_name_list=[vgg_layer],
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=False
        ).to(device)
        self.vgg.eval()

    @staticmethod
    def _phase_cosine_sim(feat1, feat2):
        F1 = torch.fft.fft2(feat1, dim=(-2, -1))
        F2 = torch.fft.fft2(feat2, dim=(-2, -1))
        p1 = torch.angle(F1)
        p2 = torch.angle(F2)
        p1f = p1.view(p1.size(0), -1)
        p2f = p2.view(p2.size(0), -1)
        return F.cosine_similarity(p1f, p2f, dim=1)

    def forward(self, pred1, pred2, target):
        """
        pred1, pred2, target: Tensor[B, C, H, W], values normalized [0,1]
        """
        B, C, H, W = pred1.shape
        mask = torch.zeros((B, 1, H, W), device=pred1.device)

        # 遍历每个样本并按块处理
        for b in range(B):
            for y in range(0, H, self.block_size):
                for x in range(0, W, self.block_size):
                    if y + self.block_size > H or x + self.block_size > W:
                        continue
                    p1 = pred1[b:b+1, :, y:y+self.block_size, x:x+self.block_size]
                    tgt = target[b:b+1, :, y:y+self.block_size, x:x+self.block_size]

                    # 提取 VGG 特征
                    f1 = self.vgg(p1)[list(self.vgg.layer_name_list)[0]]
                    f2 = self.vgg(tgt)[list(self.vgg.layer_name_list)[0]]

                    # 计算 phase cosine similarity
                    sim = self._phase_cosine_sim(f1, f2).item()
                    val = 1.0 if sim >= self.cosine_threshold else 0.0
                    mask[b, :, y:y+self.block_size, x:x+self.block_size] = val

        # 应用掩码到 pred2 和 target
        pred2_masked = pred2 * mask
        target_masked = target * mask

        # 计算 L1 损失（按掩码区域）
        l1 = F.l1_loss(pred2_masked, target_masked, reduction='sum')
        norm = mask.sum()
        loss = l1 / (norm + 1e-6)
        return loss
