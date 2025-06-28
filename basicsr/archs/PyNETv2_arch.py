import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True,
                 instance_norm=False, padding=True):

        super(ConvLayer, self).__init__()
        self.padding = padding
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        if self.padding:
            out = self.reflection_pad(x)
        else:
            out = x
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class DWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(DWConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=in_channels)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.PReLU()

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class CAMBlock(nn.Module):
    def __init__(self, in_channels, planes, instance_norm=False):
        super(CAMBlock, self).__init__()
        self.conv_l1 = ConvLayer(in_channels, planes, 3, 1, relu=True, instance_norm=instance_norm)
        self.conv_l2 = ConvLayer(planes, planes, 1, 3, relu=True, instance_norm=False)
        self.conv_l3 = ConvLayer(planes, planes, 3, 3, relu=True, instance_norm=False)
        self.conv_l4 = ConvLayer(planes, planes, 1, 1, relu=True, instance_norm=False)
        self.conv_l5 = ConvLayer(planes, planes, 1, 1, relu=False, instance_norm=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_l1(x)
        c_a = self.conv_l2(out)
        c_a = self.conv_l3(c_a)
        c_a = torch.mean(c_a, dim=[2, 3], keepdim=True)
        c_a = self.conv_l4(c_a)
        c_a = self.conv_l5(c_a)
        c_a = self.sigmoid(c_a)

        return c_a * out


class SAMBlock(nn.Module):
    def __init__(self, in_channels, planes, instance_norm):
        super(SAMBlock, self).__init__()
        self.conv_l1 = ConvLayer(in_channels, planes, 3, 1, relu=True, instance_norm=instance_norm)
        self.conv_l2 = DWConvLayer(planes, planes, 5, 1, relu=False, instance_norm=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_l1(x)
        s_a = self.conv_l2(out)
        s_a = self.sigmoid(s_a)

        return s_a * out


class Conv_1x1(nn.Module):
    def __init__(self, in_channels, planes, instance_norm):
        super(Conv_1x1, self).__init__()
        self.conv_3a = ConvLayer(in_channels, planes, 3, 1, relu=True, instance_norm=instance_norm)
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv_3a(x)
        out = self.relu(out)

        return out + x


class ConvGroups(nn.Module):
    def __init__(self, in_channels, planes, instance_norm, groups=4, n=1):
        super(ConvGroups, self).__init__()
        self.groups = groups
        self.n = n
        convs = [ConvLayer(planes // groups, planes // groups, 3, 1, relu=True,
                           instance_norm=(instance_norm and (i % 2) == 0))
                 for _ in range(n)
                 for i in range(groups)]
        self.convs = nn.Sequential(*convs)

        self.conv = ConvLayer(planes, planes, 1, 1, relu=False, instance_norm=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        split_tensors = torch.chunk(x, self.groups, dim=1)
        split_tensors_list = list(split_tensors)
        values = []
        i = 0
        for tensor in split_tensors_list:
            for _ in range(self.n):
                g = self.convs[i](tensor)
                i = i + 1
            values.append(g)
        out_cat = torch.cat(values, dim=1)
        out = self.conv(out_cat)
        return out + x

class UpConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, relu=True):

        super(UpConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)

        if relu:
            self.relu = nn.PReLU()

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out

@ARCH_REGISTRY.register()
class PyNETv2(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 level, instance_norm=True, instance_norm_level_1=False):
        super(PyNETv2, self).__init__()
        k = upscale
        self.level = level
        self.conv_l1_d1 = ConvLayer(num_in_ch, 32, 3, 1, relu=True, instance_norm=False)
        self.conv_l2_d1 = ConvLayer(32, 64, 2, 2, relu=True, instance_norm=False, padding=False)
        self.conv_l3_d1 = ConvLayer(64, 128, 2, 2, relu=True, instance_norm=False, padding=False)
        self.conv_l3_d6 = ConvGroups(128, 128, instance_norm=True, groups=4, n=2)
        self.conv_l3_d8 = SAMBlock(128, 128, instance_norm=False)
        self.conv_l3_d9 = CAMBlock(128, 128, instance_norm=False)

        self.conv_t2a = UpConvLayer(128, 64, 3, 2)
        self.conv_l2_d2 = ConvLayer(64, 64, 3, 1, relu=True, instance_norm=False)
        self.conv_l2_d12 = Conv_1x1(64, 64, instance_norm=False)
        self.conv_l2_d13 = ConvGroups(64, 64, instance_norm=True, groups=2, n=3)
        self.conv_l2_d16 = CAMBlock(64, 64, instance_norm=False)

        self.conv_t1a = UpConvLayer(64, 32, 3, 2)
        self.conv_l1_d2 = ConvLayer(32, 32, 3, 1, relu=True, instance_norm=False)
        self.conv_l1_d12 = Conv_1x1(32, 32, instance_norm=False)
        self.conv_l1_d13 = ConvGroups(32, 32, instance_norm=True, groups=4)
        self.conv_l1_d14 = ConvGroups(32, 32, instance_norm=True, groups=2)

        # -> Output: Level 1
        self.conv_l1_out = ConvLayer(32, 4 * k * k, 1, 1, relu=True, instance_norm=False)
        self.suffle3 = nn.PixelShuffle(k)

        self.conv_l0_out = ConvLayer(4, num_out_ch, 3, 1, relu=False, instance_norm=False)

    def level_1(self, conv_l1_d14):
        conv_l1_out = self.conv_l1_out(conv_l1_d14)
        conv_l1_out = self.suffle3(conv_l1_out)
        conv_l0_out = self.conv_l0_out(conv_l1_out)
        output_l1 = torch.tanh(conv_l0_out) * 0.58 + 0.5

        return output_l1
    
    def forward(self, x, noise):
        conv_l1_d1 = self.conv_l1_d1(x)

        conv_l2_d1 = self.conv_l2_d1(conv_l1_d1)
        conv_l3_d1 = self.conv_l3_d1(conv_l2_d1)
        conv_l3_d6 = self.conv_l3_d6(conv_l3_d1)
        conv_l3_d8 = self.conv_l3_d8(conv_l3_d6) + conv_l3_d6
        conv_l3_d9 = self.conv_l3_d9(conv_l3_d8) + conv_l3_d8

        conv_t2a = self.conv_t2a(conv_l3_d9)
        conv_l2_d2 = self.conv_l2_d2(conv_l2_d1)
        conv_l2_d3 = conv_l2_d2 + conv_t2a
        conv_l2_d12 = self.conv_l2_d12(conv_l2_d3)
        conv_l2_d13 = self.conv_l2_d13(conv_l2_d12)
        conv_l2_d16 = self.conv_l2_d16(conv_l2_d13) + conv_l2_d13

        conv_t1a = self.conv_t1a(conv_l2_d16)
        conv_l1_d2 = self.conv_l1_d2(conv_l1_d1)
        conv_l1_d3 = conv_l1_d2 + conv_t1a
        conv_l1_d12 = self.conv_l1_d12(conv_l1_d3)
        conv_l1_d13 = self.conv_l1_d13(conv_l1_d12)
        conv_l1_d14 = self.conv_l1_d14(conv_l1_d13)

        output_l1 = self.level_1(conv_l1_d14)

        return output_l1

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # PyNET
    net = PyNETv2(upscale=2, num_in_ch=4, num_out_ch=3, task='lsr', level=1)
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())
