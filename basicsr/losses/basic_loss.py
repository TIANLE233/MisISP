import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from basicsr.metrics.psnr_ssim import calculate_ssim_pt, calculate_psnr_pt


_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@weighted_loss
def ssim_loss(pred, target):
    return 1 - calculate_ssim_pt(pred, target, crop_border=0)

@weighted_loss
def psnr_loss(pred, target):
    return 1 - calculate_psnr_pt(pred, target, crop_border=0) * 0.01

@weighted_loss
def cx_loss(pred, target, sigma=float(1.0)):
    return CX_loss(pred, target, sigma=sigma)


@LOSS_REGISTRY.register()
class OutlierAwareLoss(nn.Module):
    def __init__(self, kernel_size=False, loss_weight=1.0, reduction='mean'):
        super(OutlierAwareLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.unfold = torch.nn.Unfold(kernel_size)
        self.kernel = kernel_size

    def forward(self, out, lab):
        b, c, h, w = out.shape
        p = self.kernel // 2
        delta = out - lab
        if self.kernel:
            delta_ = torch.nn.functional.pad(delta, (p, p, p, p))
            patch = self.unfold(delta_).reshape(b, c,
                                                self.kernel, self.kernel,
                                                h, w).detach()
            var = patch.std((2, 3)) / (2 ** .5)
            avg = patch.mean((2, 3))
        else:
            var = delta.std((2, 3), keepdims=True) / (2 ** .5)
            avg = delta.mean((2, 3), True)
        weight = 1 - (-((delta - avg).abs() / var)).exp().detach()
        # weight = 1 - (-1 / var.detach()-1/ delta.abs().detach()).exp()
        loss = (delta.abs() * weight).mean()
        return self.loss_weight * loss
    
@LOSS_REGISTRY.register()
class LossISP(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LossISP, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean())
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * ssim_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * psnr_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)
        
@LOSS_REGISTRY.register()
class patchL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', pool_size=2):
        super(patchL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.pool_size = pool_size
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        pred_pooled = F.avg_pool2d(pred, self.pool_size)
        target_pooled = F.avg_pool2d(target, self.pool_size)
        return self.loss_weight * l1_loss(pred_pooled, target_pooled, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss
@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    """
    颜色一致性损失 (L_col)
    作用：约束生成图像中 R、G、B 三个通道的平均值接近，从而实现色彩平衡。
    """
    def __init__(self):
        super(ColorLoss, self).__init__()
    
    def forward(self, J: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算颜色一致性损失

        参数：
            J: torch.Tensor, 输入图像张量，形状为 [batch_size, 3, height, width]
        
        返回：
            L_col: torch.Tensor, 颜色一致性损失，标量张量
        """
        # 检查输入是否为 3 通道
        assert J.shape[1] == 3, "输入图像必须为 3 通道 (RGB)"
        
        # 计算每个通道的平均值
        J_R = torch.mean(J[:, 0, :, :])  # R 通道均值
        J_G = torch.mean(J[:, 1, :, :])  # G 通道均值
        J_B = torch.mean(J[:, 2, :, :])  # B 通道均值
        
        # 计算通道间差异的平方
        diff_RG = (J_R - J_G) ** 2
        diff_GB = (J_G - J_B) ** 2
        diff_BR = (J_B - J_R) ** 2
        
        # 计算颜色一致性损失
        L_col = torch.sqrt(diff_RG + diff_GB + diff_BR)
        
        return L_col


@LOSS_REGISTRY.register()
class CXLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', sigma=float(1.0)):
        super(CXLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sigma = sigma
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * cx_loss(pred, target, weight, sigma=float(self.sigma), reduction=self.reduction)

@LOSS_REGISTRY.register()
class PerceptualCXLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', sigma=float(1.0)):
        super(PerceptualCXLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sigma = sigma
        layer_weights={'conv4_2': 1.}

        self.vgg = VGGFeatureExtractor(
        layer_name_list=list(layer_weights.keys()),
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        pred_features = self.vgg(pred)
        gt_features = self.vgg(target.detach())

        # self.loss_content= 
        percep_loss = 0
        for k in pred_features.keys():
            percep_loss += cx_loss(pred_features[k], gt_features[k], 
                                   weight, sigma=float(self.sigma), reduction=self.reduction) * self.layer_weights[k]

        return percep_loss * self.loss_weight
        # return self.loss_weight * cx_loss(pred, target, weight, sigma=float(self.sigma), reduction=self.reduction)

@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
