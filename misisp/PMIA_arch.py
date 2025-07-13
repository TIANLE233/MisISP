import yaml
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torchvision.transforms as T

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),

            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 19)  # 输出19个参数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class PMIA_Encoder(nn.Module):
    def __init__(self):
        super(PMIA_Encoder, self).__init__()
        self.encoder = Encoder() 
        self.decoder = Decoder()  

        # 添加全局平均池化层
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, gt: torch.Tensor) -> torch.Tensor:
        # 经过编码器提取特征
        x_re = F.interpolate(x, size=512, mode='bilinear', align_corners=False)
        gt_re = F.interpolate(gt, size=512, mode='bilinear', align_corners=False)
        y = torch.cat([x_re, gt_re], dim=1)

        y = self.encoder(y)

        # 对特征图进行全局平均池化，输出每个通道的均值
        y = self.pool(y)

        # 将输出展平为 1D 向量
        y = y.view(x.size(0), -1)  

        # 经过解码器
        y = self.decoder(y)
        
        return y


class ISPFunctions(nn.Module):
    # def __init__(self, output_dir='./isp_parameters'):
    def __init__(self):
        super(ISPFunctions, self).__init__()
        # 初始化参数直接定义在ISPFunctions中；好的初始化很重要
        self.phi_dg_init = 0.8
        self.phi_wb_r_init = 1.0
        self.phi_wb_b_init = 1.0
        self.phi_ccm_init = torch.eye(3).flatten()
        self.phi_o_init = torch.zeros(3)
        self.phi_gamma_init = 0.25
        self.phi_s_init = 3
        self.phi_p1_init = 1
        self.phi_p2_init = 1
        # self.phi_r1_init = 1  # 主轴半径，形状为 (B,)，取1
        # self.phi_r2_init = 1  # 次轴半径，形状为 (B,)，取1
        # self.phi_theta_init = 0  # 高斯核角度，形状为 (B,)，取0
        # self.phi_sharpen_init = 1

    def forward(self, x: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        # 计算最终参数
        final_params = self.calculate_final_params(residuals)

        # 参数解包
        phi_dg = final_params[:, 0]
        phi_r, phi_b = final_params[:, 1], final_params[:, 2]
        phi_ccm = final_params[:, 3:12].view(-1, 3, 3)
        phi_o = final_params[:, 12:15]
        phi_gamma = final_params[:, 15]
        phi_s = final_params[:, 16]
        phi_p1 = final_params[:, 17]
        phi_p2 = final_params[:, 18]
        # phi_r1 = final_params[:, 19]  
        # phi_r2 = final_params[:, 20]   
        # phi_theta = final_params[:, 21]  
        # phi_sharpen = final_params[:, 22] 

        # 应用各种ISP功能
        # x = bayer_demosaic_torch(x)
        x = self.digital_gain(x, phi_dg)
        x = self.white_balance(x, phi_r, phi_b)
        # x_o = x.clone()
        # x = self.denoise(x, phi_r1, phi_r2, phi_theta)
        # x = self.sharpen(x, x_o, phi_sharpen)
        x = self.color_correction(x, phi_ccm, phi_o)
        x = self.gamma_correction(x, phi_gamma)
        x = self.tone_mapping(x, phi_s, phi_p1, phi_p2)

        return x

    def calculate_final_params(self, residuals):
        # 使用内部定义的初始化参数
        init_params_tensor = torch.tensor([
            self.phi_dg_init,
            self.phi_wb_r_init,
            self.phi_wb_b_init,
            *self.phi_ccm_init,
            *self.phi_o_init,
            self.phi_gamma_init,
            self.phi_s_init,
            self.phi_p1_init,
            self.phi_p2_init,
            # self.phi_r1_init,
            # self.phi_r2_init,
            # self.phi_theta_init,
            # self.phi_sharpen_init
        ], dtype=torch.float32)

        # 广播初始化参数以匹配残差的批处理维度
        init_params_batched = init_params_tensor.unsqueeze(0).expand(residuals.size(0), -1).to(residuals.device)

        # 返回批处理的最终参数
        return init_params_batched + residuals

    # 定义具体的ISP功能处理方法
    def digital_gain(self, x: torch.Tensor, phi_dg: torch.Tensor) -> torch.Tensor:
        phi_dg = torch.clamp(phi_dg, min=0.5, max=10)
        output = phi_dg[:, None, None, None] * x 
        return torch.clamp(output, min=0.0, max=1.0)
        
    def white_balance(self, x: torch.Tensor, phi_r: torch.Tensor, phi_b: torch.Tensor) -> torch.Tensor:
        x_r, x_g, x_b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        phi_r = torch.clamp(phi_r, min=0.1, max=5)
        phi_b = torch.clamp(phi_b, min=0.1, max=5)
        x_r = phi_r[:, None, None] * x_r
        x_b = phi_b[:, None, None] * x_b
        output = torch.stack((x_r, x_g, x_b), dim=1)
        return torch.clamp(output, min=0.0, max=1.0)

    def color_correction(self, x: torch.Tensor, phi_ccm: torch.Tensor, phi_o: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), 3, -1)
        x_cc = torch.bmm(phi_ccm, x_flat) + phi_o.unsqueeze(2)
        output = x_cc.view_as(x)
        return torch.clamp(output, min=0.0, max=1.0)

    def gamma_correction(self, x: torch.Tensor, phi_gamma: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8
        phi_gamma = torch.clamp(phi_gamma, min=0.1, max=5)
        output = torch.pow(torch.clamp(x, min=epsilon), phi_gamma[:, None, None, None])
        return torch.clamp(output, min=0.0, max=1.0)

    def tone_mapping(self, x: torch.Tensor, phi_s: torch.Tensor, phi_p1: torch.Tensor, phi_p2: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8  # 定义极小值以避免计算中的数值问题
        phi_p1 = torch.clamp(phi_p1, min=0.1, max=5)
        phi_p2 = torch.clamp(phi_p2, min=0.1, max=5)
        phi_s = torch.clamp(phi_s, min=1, max=10)
        x_p1 = torch.pow(torch.clamp(x, min=epsilon), phi_p1[:, None, None, None])
        x_p2 = torch.pow(torch.clamp(x, min=epsilon), phi_p2[:, None, None, None])
        output = phi_s[:, None, None, None] * x_p1 - (phi_s[:, None, None, None] - 1) * x_p2
        return torch.clamp(output, min=0.0, max=1.0)



    # def denoise(self, x: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, theta: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    #     """
    #     生成自适应高斯核并应用于去噪
    #     :param x: 输入图像，形状为 (B, C, H, W)
    #     :param r1: 主轴半径，形状为 (B,)
    #     :param r2: 次轴半径，形状为 (B,)
    #     :param theta: 高斯核角度，形状为 (B,)
    #     :param kernel_size: 高斯核的大小 (默认5x5)
    #     :return: 去噪后的图像
    #     """
    #     # 获取输入图像所在的设备
    #     device = x.device

    #     # 获取批次大小
    #     batch_size = x.size(0)
    #     channels = x.size(1)

    #     # 初始化一个列表来存储每张图像的去噪结果
    #     denoised_images = []

    #     # 对批次中的每一张图像单独进行去噪
    #     for i in range(batch_size):
    #         # 提取当前图像
    #         current_image = x[i:i+1]  # (1, C, H, W)

    #         # 获取当前图像对应的 r1, r2 和 theta
    #         r1_current = r1[i].unsqueeze(0)  # (1,)
    #         r2_current = r2[i].unsqueeze(0)  # (1,)
    #         theta_current = theta[i].unsqueeze(0)  # (1,)

    #         # 生成自适应高斯核
    #         r1_current = torch.clamp(r1_current, min=1e-6, max=5).to(device)
    #         r2_current = torch.clamp(r2_current, min=1e-6, max=5).to(device)
    #         theta_current = torch.clamp(theta_current, min=0, max=2 * torch.pi).to(device)

    #         coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    #         x_coords, y_coords = torch.meshgrid(coords, coords, indexing="ij")
    #         x_coords, y_coords = x_coords[None, :, :], y_coords[None, :, :]

    #         b0 = (torch.cos(theta_current) ** 2) / (2 * r1_current ** 2) + (torch.sin(theta_current) ** 2) / (2 * r2_current ** 2)
    #         b1 = (torch.sin(2 * theta_current)) / (4 * r1_current ** 2) * ((r1_current / r2_current) ** 2 - 1)
    #         b2 = (torch.sin(theta_current) ** 2) / (2 * r1_current ** 2) + (torch.cos(theta_current) ** 2) / (2 * r2_current ** 2)

    #         # 计算高斯核
    #         kernel = torch.exp(- (b0[:, None, None] * x_coords ** 2 + 2 * b1[:, None, None] * x_coords * y_coords + b2[:, None, None] * y_coords ** 2))
    #         kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)  # 归一化
    #         kernel = kernel.unsqueeze(1)  # 添加通道维度，使其形状为 (1, 1, size, size)

    #         # 对每个通道分别应用相同的高斯核
    #         denoised_channels = []
    #         for c in range(channels):
    #             # 对当前通道进行卷积
    #             denoised_channel = F.conv2d(current_image[:, c:c+1], kernel, padding=kernel_size // 2)
    #             denoised_channels.append(denoised_channel)

    #         # 将所有通道的去噪结果合并
    #         denoised_image = torch.cat(denoised_channels, dim=1)  # (1, C, H, W)
    #         denoised_images.append(denoised_image)

    #     # 将所有批次的去噪图像合并成一个张量
    #     denoised_images = torch.cat(denoised_images, dim=0)

    #     return denoised_images


    # def sharpen(self, x: torch.Tensor, orign: torch.Tensor, phi_sharpen: torch.Tensor) -> torch.Tensor:
    #     """
    #     锐化图像
    #     :param x: 输入图像，形状为 (B, C, H, W)
    #     :param denoised: 去噪后的图像，形状为 (B, C, H, W)
    #     :param phi_sharpen: 锐化参数，形状为 (B,)
    #     :return: 锐化后的图像
    #     """
    #     phi_sharpen = torch.clamp(phi_sharpen, min=0, max=2)
    #     residual = orign - x
    #     return x + phi_sharpen[:, None, None, None] * residual


@ARCH_REGISTRY.register()
class PMIA(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str):
        super(PMIA, self).__init__()
        self.style_encoder = PMIA_Encoder()
        self.isp_functions = ISPFunctions()

    def forward(self, x, gt: torch.Tensor) -> torch.Tensor:
        # 获取Style_Encoder的输出参数
        residuals = self.style_encoder(x, gt)
        
        # 将这些参数作为ISPFunctions的输入
        output = self.isp_functions(x, residuals)
        
        return output

