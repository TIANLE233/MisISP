import cv2
import torch

from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@LOSS_REGISTRY.register()
class ADLossall(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', layer_weights=None, amp_weight=1.0, phase_weight=0.0, use_fft=False, patch_size=5, stride=1, num_proj=256):
        """
        ADLoss: 利用VGG特征并结合随机投影，计算输入图像之间的特征差异，并可选择性地在VGG之前进行模糊操作或加入FFT相位计算。
        
        Args:
            loss_weight (float): 损失权重。
            reduction (str): 损失的归约方式，可选 'none', 'mean', 'sum'。
            layer_weights (dict): 指定使用的VGG层及其对应的权重，例如 {'conv4_2': 1.0}。
            amp_weight (float): 幅值差异的权重。
            phase_weight (float): 相位差异的权重。
            use_fft (bool): 是否在计算损失时使用FFT幅值和相位。
            patch_size (int): 随机投影的patch大小。
            stride (int): 随机投影的步幅。
            num_proj (int): 随机投影的数量。
            blur_kernel_size (int): 模糊核大小，若为0则不进行模糊。
        """
        super(ADLossall, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported modes are: none, mean, sum')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        self.use_fft = use_fft
        self.stride = stride
        self.layer_weights = layer_weights if layer_weights else {'conv4_2': 1.0}

        # 定义VGG各层的输出通道数
        self.layer_channels = {
            'conv1_2': 64,
            'conv2_2': 128,
            'conv3_2': 256,
            'conv4_2': 512,
            'conv5_2': 512
        }

        # 创建VGG特征提取器
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(self.layer_weights.keys()),  
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=False)

        # 随机投影初始化
        for layer in self.layer_weights.keys():
            chn = self.layer_channels[layer]
            rand = torch.randn(num_proj, chn, patch_size, patch_size)
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f'rand_{layer}', rand)

    def forward_once(self, x, y, layer):
        """
        计算输入图像x与y在指定层的特征差异
        """
        proj_kernel = self.__getattr__(f'rand_{layer}')
        projx = F.conv2d(x, proj_kernel, stride=self.stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, proj_kernel, stride=self.stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # 对投影后的特征进行排序
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # 计算排序后特征的差异（幅值差异）
        s = torch.abs(projx - projy).mean([1, 2])

        return s

    def forward(self, pred, target):
        # 提取预测图像和目标图像的VGG特征
        pred_features = self.vgg(pred)
        target_features = self.vgg(target.detach())

        score = 0  # 初始化总损失
        for layer in self.layer_weights.keys():
            pred_layer_features = pred_features[layer]
            target_layer_features = target_features[layer]

            if self.use_fft:
                # 使用FFT计算幅值和相位
                fft_pred = torch.fft.fftn(pred_layer_features, dim=(-2, -1))
                fft_target = torch.fft.fftn(target_layer_features, dim=(-2, -1))

                # 计算幅值和相位
                pred_amp = torch.abs(fft_pred)
                pred_phase = torch.angle(fft_pred)
                target_amp = torch.abs(fft_target)
                target_phase = torch.angle(fft_target)

                # 计算幅值差异
                s_amplitude = self.forward_once(pred_amp, target_amp, layer)
                score += s_amplitude * self.layer_weights[layer] * self.amp_weight

                # 计算相位差异
                if self.phase_weight > 0:
                    s_phase = self.forward_once(pred_phase, target_phase, layer)
                    score += s_phase * self.layer_weights[layer] * self.phase_weight

            else:
                # 直接计算SWD差异
                s_amplitude = self.forward_once(torch.abs(pred_layer_features), torch.abs(target_layer_features), layer)
                score += s_amplitude * self.layer_weights[layer] * self.amp_weight

        # 根据reduction方式计算最终损失
        if self.reduction == 'mean':
            score = score.mean()
        elif self.reduction == 'sum':
            score = score.sum()

        return score * self.loss_weight
