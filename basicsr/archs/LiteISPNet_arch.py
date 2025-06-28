import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.style_arch import StyleModel, Encoder
# from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''


def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])


'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''


# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='zeros', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation,
                                        padding_mode=padding_mode))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'S':
            L.append(nn.Sigmoid())
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return seq(*L)


class DWTForward(nn.Conv2d):
    def __init__(self, in_channels=64):
        super(DWTForward, self).__init__(in_channels, in_channels * 4, 2, 2,
                                         groups=in_channels, bias=False)
        weight = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]],
                               [[[0.5, 0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5, 0.5]]]],
                              dtype=torch.get_default_dtype()
                              ).repeat(in_channels, 1, 1, 1)  # / 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)


class DWTInverse(nn.ConvTranspose2d):
    def __init__(self, in_channels=64):
        super(DWTInverse, self).__init__(in_channels, in_channels // 4, 2, 2,
                                         groups=in_channels // 4, bias=False)
        weight = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]],
                               [[[0.5, 0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5, 0.5]]]],
                              dtype=torch.get_default_dtype()
                              ).repeat(in_channels // 4, 1, 1, 1)  # * 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding=padding, bias=bias, mode=mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
                       bias, mode, reduction) for _ in range(nb)]
        # RG = [ResBlock(in_channels, out_channels, kernel_size, stride, padding,
        #                bias, mode) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))

        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x


@ARCH_REGISTRY.register()
class LiteISPNet(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                ):
        super(LiteISPNet, self).__init__()

        ch_1 = 64
        ch_2 = 128
        ch_3 = 128
        n_blocks = 4

    
        self.head = seq(
            conv(4, ch_1, mode='C')
        )  # shape: (N, ch_1, H/2, W/2)

        self.down1 = seq(
            conv(ch_1, ch_1, mode='C'),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            conv(ch_1, ch_1, mode='C'),
            DWTForward(ch_1)
        )  # shape: (N, ch_1*4, H/4, W/4)

        self.down2 = seq(
            conv(ch_1 * 4, ch_1, mode='C'),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            DWTForward(ch_1)
        )  # shape: (N, ch_1*4, H/8, W/8)

        self.down3 = seq(
            conv(ch_1 * 4, ch_2, mode='C'),
            RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            DWTForward(ch_2)
        )  # shape: (N, ch_2*4, H/16, W/16)

        self.middle = seq(
            conv(ch_2 * 4 , ch_3, mode='C'),
            RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            conv(ch_3, ch_2 * 4, mode='C')
        )  # shape: (N, ch_2*4, H/16, W/16)

        self.up3 = seq(
            DWTInverse(ch_2 * 4),
            RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            conv(ch_2, ch_1 * 4, mode='C')
        )  # shape: (N, ch_1*4, H/8, W/8)

        self.up2 = seq(
            DWTInverse(ch_1 * 4),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            conv(ch_1, ch_1 * 4, mode='C')
        )  # shape: (N, ch_1*4, H/4, W/4)

        self.up1 = seq(
            DWTInverse(ch_1 * 4),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            conv(ch_1, ch_1, mode='C')
        )  # shape: (N, ch_1, H/2, W/2)

        self.tail = seq(
            conv(ch_1, ch_1 * 4, mode='C'),
            nn.PixelShuffle(upscale_factor=2),
            conv(ch_1, 3, mode='C')
        )  # shape: (N, 3, H, W)

    def forward(self, raw):
        # # input = raw
        # N, C, H, W = raw.shape
        # input = torch.pow(raw, 1 / 2.2)
        h = self.head(raw)

        d1 = self.down1(h)
        d2 = self.down2(d1)
        d3 = self.down3(d2) 

        m = self.middle(d3) + d3
        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2) + h
        out = self.tail(u1)

        return out

@ARCH_REGISTRY.register()
class LiteISPNet_noise(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                ):
        super(LiteISPNet_noise, self).__init__()

        ch_1 = 64
        ch_2 = 128
        ch_3 = 128
        n_blocks = 4

        self.head = seq(
            conv(4, ch_1 - 1, mode='C')
        )  # shape: (N, ch_1, H/2, W/2)

        self.down1 = seq(
            conv(ch_1, ch_1, mode='C'),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            conv(ch_1, ch_1, mode='C'),
            DWTForward(ch_1)
        )  # shape: (N, ch_1*4, H/4, W/4)

        self.down2 = seq(
            conv(ch_1 * 4, ch_1, mode='C'),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            DWTForward(ch_1)
        )  # shape: (N, ch_1*4, H/8, W/8)

        self.down3 = seq(
            conv(ch_1 * 4, ch_2, mode='C'),
            RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            DWTForward(ch_2)
        )  # shape: (N, ch_2*4, H/16, W/16)

        self.middle = seq(
            conv(ch_2 * 4, ch_3, mode='C'),
            RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            conv(ch_3, ch_2 * 4, mode='C')
        )  # shape: (N, ch_2*4, H/16, W/16)

        self.up3 = seq(
            DWTInverse(ch_2 * 4),
            RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            conv(ch_2, ch_1 * 4, mode='C')
        )  # shape: (N, ch_1*4, H/8, W/8)

        self.up2 = seq(
            DWTInverse(ch_1 * 4),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            conv(ch_1, ch_1 * 4, mode='C')
        )  # shape: (N, ch_1*4, H/4, W/4)

        self.up1 = seq(
            DWTInverse(ch_1 * 4),
            RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            conv(ch_1, ch_1, mode='C')
        )  # shape: (N, ch_1, H/2, W/2)

        self.tail = seq(
            conv(ch_1, ch_1 * 4, mode='C'),
            nn.PixelShuffle(upscale_factor=2),
            conv(ch_1, 3, mode='C')
        )  # shape: (N, 3, H, W)

    def forward(self, raw, noise_tensor):

        h = self.head(raw)
        B, C, H, W = h.shape
        noise_map = noise_tensor.view(B,1,1,1).expand(B, 1, H, W)
        h_noise = torch.cat([h, noise_map], dim=1)

        d1 = self.down1(h_noise)
        d2 = self.down2(d1)
        d3 = self.down3(d2) 

        m = self.middle(d3) + d3
        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2) + h_noise
        out = self.tail(u1)

        return out 

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = LiteISPNet()
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())
