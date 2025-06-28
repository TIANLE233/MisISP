import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class DoubleAttention(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', mid_activation='none', end_activation='sigmoid',
                 norm='none', reduction=1, multiplier=1):
        super(DoubleAttention, self).__init__()
        self.mid_activation = mid_activation
        self.conv1 = nn.Conv2d(in_channels, out_channels * multiplier, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels * multiplier, out_channels * multiplier, kernel_size=1, stride=1, padding=0)
        self.ca = ChannelAttention(out_channels * multiplier, activation, end_activation, reduction)
        self.sa = SpatialAttention(out_channels * multiplier, end_activation)
        self.conv3 = nn.Conv2d(out_channels * 2 * multiplier, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = norm

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)  # Assuming lrelu is used as activation
        x = self.conv2(x)
        x = F.leaky_relu(x) if self.mid_activation == 'lrelu' else x  # Assuming lrelu is used as activation
        ca = self.ca(x)
        sa = self.sa(x)
        x = torch.cat((ca, sa), dim=1)
        x = self.conv3(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, activation='relu', end_activation='sigmoid', reduction=1):
        self.end_activation = end_activation
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = F.leaky_relu(y)  # Assuming lrelu is used as activation
        y = self.conv2(y)
        y = torch.sigmoid(y) if self.end_activation == 'sigmoid' else y
        return y * x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, end_activation='sigmoid'):
        super(SpatialAttention, self).__init__()
        self.end_activation = end_activation
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1,
                              padding=4, dilation=2,  groups=in_channels)

    def forward(self, x):
        y = self.dwconv(x)
        y = torch.sigmoid(y) if self.end_activation == 'sigmoid' else y
        return y * x


@ARCH_REGISTRY.register()
class LAN(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 activation='lrelu', end_activation='tanh'):
        super(LAN, self).__init__()
        self.activation = activation
        self.end_activation = end_activation
        self.conv1 = nn.Conv2d(num_in_ch, 16, kernel_size=4, stride=2, padding=1)
        self.downscale = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.dam1 = DoubleAttention(16, 16, activation=self.activation, mid_activation='none',
                                    end_activation=self.end_activation, reduction=4, multiplier=2)
        self.dam2 = DoubleAttention(16, 16, activation=self.activation, mid_activation='none',
                                    end_activation=self.end_activation, reduction=4, multiplier=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.upscale1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.upscale2 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv5 = nn.Conv2d(64, num_out_ch, kernel_size=3, stride=1, padding=1)
        # self.up = nn.PixelShuffle(upscale)
            # 第三层卷积，输入通道16，输出通道12，使用ReLU激活函数
        self.tail = nn.Sequential(nn.Conv2d(64, num_out_ch * (upscale**2), kernel_size=(3, 3), padding=1),
                                #   nn.ReLU(),\
                                  nn.PixelShuffle(upscale))

    def forward(self, x):
        x_a = F.leaky_relu(self.conv1(x))
        x_b = F.leaky_relu(self.downscale(x_a))
        x_c = F.leaky_relu(self.conv2(x_b))

        dam1 = self.dam1(x_c)
        dam1 = x_c + dam1

        dam2 = self.dam2(dam1)
        dam2 = dam1 + dam2

        y = F.leaky_relu(self.conv3(dam2))
        y = x_c + y

        z = self.upscale1(y)
        z = torch.cat((x_a, z), dim=1)
        z = F.leaky_relu(self.conv4(z))
        z = self.upscale2(z)
        z = F.leaky_relu(self.tail(z))
        out = torch.tanh(z) * 0.58 + 0.5  # Assuming tanh is used as end_activation
        return out


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # PyNET
    net = LAN()
    print(count_parameters(net))

    data = torch.randn(1, 3, 224, 224)
    print(net(data).size())
