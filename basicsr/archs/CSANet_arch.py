import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class CSANet(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str, instance_norm=True):
        super(CSANet, self).__init__()
        self.conv1 = ConvMultiBlock(num_in_ch, 32, kernel_size=5, instance_norm=True)
        self.out_att1 = AttentionModule(32, ratio=2, kernel_size=3)
        self.out_att2 = AttentionModule(32, ratio=2, kernel_size=3)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=1, relu=False)
        self.transpose_conv = nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = ConvLayer(64, 12, kernel_size=3, stride=1, relu=False)
        self.pix_shuff = nn.PixelShuffle(2)
        self.conv4 = ConvLayer(3, 3, kernel_size=3, stride=1, instance_norm=False, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        z1 = self.out_att1(conv1) + conv1
        z2 = self.out_att2(z1) + z1
        conv2 = self.conv2(z2)
        z3 = torch.cat([conv2, conv1], dim=1)
        t_conv = self.transpose_conv(z3)
        conv3 = self.conv3(t_conv)
        pix_shuff = self.pix_shuff(conv3)
        return self.sigmoid(self.conv4(pix_shuff))


class ConvMultiBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, instance_norm):
        super(ConvMultiBlock, self).__init__()
        self.conv_a = ConvLayer(in_channels, out_channels, kernel_size, stride=1, instance_norm=instance_norm)
        self.conv_b = ConvLayer(out_channels, out_channels, kernel_size, stride=1, instance_norm=instance_norm)

    def forward(self, x):
        return self.conv_b(self.conv_a(x))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True) if instance_norm else None
        self.relu = nn.LeakyReLU(0.2) if relu else None

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        if self.instance_norm:
            x = self.instance_norm(x)
        if self.relu:
            x = self.relu(x)
        return x


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        padding = 2 * (kernel_size // 2)
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, dilation=2, groups=in_channels),
            nn.ReLU()
        )
        self.point_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.dw_conv(x)
        return self.sigmoid(self.point_conv(x))


class SpatialAttention2(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(SpatialAttention2, self).__init__()
        self.dw = DepthwiseConv(in_channels, kernel_size)

    def forward(self, x):
        return x * self.dw(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(nn.AdaptiveAvgPool2d(1)(x))
        max_out = self.fc(nn.AdaptiveMaxPool2d(1)(x))
        return x * self.sigmoid(avg_out + max_out)


class AttentionModule(nn.Module):
    def __init__(self, in_channels, ratio, kernel_size):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.ca = ChannelAttention(32, ratio)
        self.sa = SpatialAttention2(32, kernel_size)
        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv2(torch.tanh(self.conv1(x)))
        ca_out = self.ca(x)
        sa_out = self.sa(x)
        return self.conv3(torch.cat([ca_out, sa_out], dim=1))


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = CSANET()
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())