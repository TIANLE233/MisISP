import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

class PredictModule(nn.Module):
    def __init__(self, in_channels, N=10):
        super(PredictModule, self).__init__()
        self.N = N

        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, N, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.gap(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        weights = self.softmax(x).unsqueeze(-1)  # (B, N, 1)
        return weights

class MCCB(nn.Module):
    def __init__(self, in_channels, N=10, K=64):
        super(MCCB, self).__init__()
        self.N = N
        self.K = K
        
        # Layers for x1
        self.ln = nn.LayerNorm(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        
        # PredictModule for the secondary branch
        self.predict_module = PredictModule(in_channels, N)
        
        # Learnable base matrices (N, K, K)
        self.base_matrices = nn.Parameter(torch.randn(N, K, K))
        
        # Layers for the main branch
        self.conv3_up = nn.Conv2d(in_channels, K, kernel_size=3, padding=1)
        self.conv3_down = nn.Conv2d(K, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_c = x  # Copy of input
        
        # Generate x1
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        x1 = self.conv3(x)
        x1 = self.gelu(x1)
        
        # Secondary branch
        weights = self.predict_module(x1)  # (B, N, 1)
        weighted_matrices = torch.einsum('bnk,nkl->bkl', weights, self.base_matrices)  # (B, K, K)
        
        # Main branch
        x_main = self.conv3_up(x1)  # (B, K, H, W)
        B, K, H, W = x_main.shape
        x_main = x_main.view(B, K, -1)  # (B, K, HW)
        x_main = torch.einsum('bkl,bkm->blm', weighted_matrices, x_main)  # (B, K, HW)
        x_main = x_main.view(B, K, H, W)  # (B, K, H, W)
        x_main = self.conv3_down(x_main)  # (B, C, H, W)
        
        # Add to x_c and output
        output = x_c + x_main
        return output
    
class MSCB(nn.Module):
    def __init__(self, in_channels, N=10, K=64):
        super(MSCB, self).__init__()
        self.N = N
        self.K = K
        
        # Layers for x1
        self.ln = nn.LayerNorm(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        
        # PredictModule for the secondary branch
        self.predict_module = PredictModule(in_channels, N)
        
        # Learnable base matrices (N, 1, K)
        self.base_matrices = nn.Parameter(torch.randn(N, 1, K))
        
        # Layers for the main branch
        self.conv3_up = nn.Conv2d(in_channels, K, kernel_size=3, padding=1)
        self.conv3_down = nn.Conv2d(K, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_c = x  # Copy of input
        
        # Generate x1
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        x1 = self.conv3(x)
        x1 = self.gelu(x1)
        
        # Secondary branch
        weights = self.predict_module(x1)  # (B, N, 1)
        weighted_vector = torch.einsum('bnk,nkl->bkl', weights, self.base_matrices)  # (B, 1, K)
        
        # Main branch
        p_c = self.conv3_up(x1)  # (B, K, H, W)
        B, K, H, W = p_c.shape
        p_c_reshaped = p_c.view(B, K, -1)  # (B, K, HW)
        
        # Compute cosine similarity between weighted_vector and p_c_reshaped
        weighted_vector_norm = F.normalize(weighted_vector, dim=2)  # (B, 1, K)
        p_c_reshaped_norm = F.normalize(p_c_reshaped, dim=1)  # (B, K, HW)
        cosine_sim = torch.einsum('bkl,bkm->bm', weighted_vector_norm, p_c_reshaped_norm)  # (B, HW)
        
        # Reshape cosine similarity to (B, 1, H, W)
        cosine_sim = cosine_sim.view(B, 1, H, W)  # (B, 1, H, W)
        
        # Multiply p_c with cosine similarity
        p_c_weighted = p_c * cosine_sim  # (B, K, H, W)
        
        # Down-sample back to (B, C, H, W)
        output = self.conv3_down(p_c_weighted)  # (B, C, H, W)
        
        # Add to x_c and output
        output = x_c + output
        return output
    
class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.mccb = MCCB(in_channels, N=10, K=40)
        self.mscb = MSCB(in_channels, N=10, K=40)

    def forward(self, x):
        x = self.mccb(x)
        x = self.mscb(x)
        return x
    
@ARCH_REGISTRY.register()
class MetaISP(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str):
        super(MetaISP, self).__init__()
        self.upscale = upscale

        # 浅层特征提取
        self.shallow_feature_extraction = nn.Conv2d(num_in_ch, 32, kernel_size=3, padding=1)

        # 降维过程
        self.down1 = nn.Sequential(
            Block(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            Block(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.down3 = nn.Sequential(
            Block(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )

        # 中间处理
        self.middle1 = Block(64)
        self.middle2 = Block(64)

        # 升维过程
        self.up3 = nn.Sequential(
            Block(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.up2 = nn.Sequential(
            Block(32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.up1 = nn.Sequential(
            Block(32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # 重建模块
        self.tail = nn.Sequential(
            nn.Conv2d(32, 32 * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(32, num_out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, n ):
        # 浅层特征提取
        h = self.shallow_feature_extraction(x)

        # 降维过程
        d1 = self.down1(h)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # 中间处理
        m = self.middle1(d3)
        m = self.middle2(m)

        # 升维过程 + 残差连接
        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2) + h

        # 重建模块
        out = self.tail(u1)
        return out

@ARCH_REGISTRY.register()
class MetaISP_noise(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str):
        super(MetaISP_noise, self).__init__()
        self.upscale = upscale

        # 浅层特征提取
        self.shallow_feature_extraction = nn.Conv2d(num_in_ch, 32-1, kernel_size=3, padding=1)

        # 降维过程
        self.down1 = nn.Sequential(
            Block(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            Block(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.down3 = nn.Sequential(
            Block(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )

        # 中间处理
        self.middle1 = Block(64)
        self.middle2 = Block(64)

        # 升维过程
        self.up3 = nn.Sequential(
            Block(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.up2 = nn.Sequential(
            Block(32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.up1 = nn.Sequential(
            Block(32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # 重建模块
        self.tail = nn.Sequential(
            nn.Conv2d(32, 32 * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(32, num_out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, noise_tensor):
        # 浅层特征提取
        h = self.shallow_feature_extraction(x)

        B, C, H, W = h.shape
        noise_map = noise_tensor.view(B,1,1,1).expand(B, 1, H, W)
        h_noise = torch.cat([h, noise_map], dim=1)

        # 降维过程
        d1 = self.down1(h_noise)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # 中间处理
        m = self.middle1(d3)
        m = self.middle2(m)

        # 升维过程 + 残差连接
        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2) + h_noise

        # 重建模块
        out = self.tail(u1)
        return out

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = MetaISP(upscale=2, num_in_ch=4, num_out_ch=3, task='lsr')
    print(count_parameters(net))

    data = torch.randn(1, 4, 120, 80)
    print(net(data).size())
