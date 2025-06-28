import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class GCRDB(nn.Module):
    def __init__(self, in_channels, att_block, num_dense_layer=6, growth_rate=16):
        super(GCRDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        self.final_att = att_block(inplanes=in_channels, planes=in_channels)

    def forward(self, x):
        out_rdb = self.residual_dense_layers(x)
        out_rdb = self.conv_1x1(out_rdb)
        out_rdb = self.final_att(out_rdb)
        out = out_rdb + x
        return out


class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm_layer = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.norm_layer(out)
        out = torch.cat((x, out), 1)
        return out


class SE_net(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, attention=True):
        super().__init__()
        self.attention = attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_mid = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, padding=0)

        self.x_red = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):

        if self.attention is True:
            y = self.avg_pool(x)
            y = F.relu(self.conv_in(y))
            y = F.relu(self.conv_mid(y))
            y = torch.sigmoid(self.conv_out(y))
            x = self.x_red(x)
            return x * y
        else:
            return x


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes=9, planes=32, pool='att', fusions=['channel_add'], ratio=4):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)  # context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class GCWTResDown(nn.Module):
    def __init__(self, in_channels, att_block, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dwt = DWT()
        if norm_layer:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                      norm_layer(in_channels),
                                      nn.PReLU(),
                                      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      norm_layer(in_channels),
                                      nn.PReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.PReLU(),
                                      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.PReLU())
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_down = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        #self.att = att_block(in_channels * 2, in_channels * 2)

    def forward(self, x):
        stem = self.stem(x)
        xLL, dwt = self.dwt(x)
        res = self.conv1x1(xLL)
        out = torch.cat([stem, res], dim=1)
        #out = self.att(out)
        return out, dwt


class GCIWTResUp(nn.Module):

    def __init__(self, in_channels, att_block, norm_layer=None):
        super().__init__()
        if norm_layer:
            self.stem = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                norm_layer(in_channels // 4),
                nn.PReLU(),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                norm_layer(in_channels // 4),
                nn.PReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                nn.PReLU(),
            )
        self.pre_conv_stem = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, padding=0)
        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        # self.prelu = nn.PReLU()
        self.post_conv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1, padding=0)
        self.iwt = IWT()
        self.last_conv = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, padding=0)
        # self.se = SE_net(in_channels // 2, in_channels // 4)

    def forward(self, x, x_dwt):
        x = self.pre_conv_stem(x)
        stem = self.stem(x)
        x_dwt = self.pre_conv(x_dwt)
        x_iwt = self.iwt(x_dwt)
        x_iwt = self.post_conv(x_iwt)
        out = torch.cat((stem, x_iwt), dim=1)
        out = self.last_conv(out)
        return out


class shortcutblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.se = SE_net(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.se(self.relu(self.conv2(self.relu(self.conv1(x)))))


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class last_upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.PixelShuffle(2)
        self.pre_enhance = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.enhance = PSPModule(16, 16)
        self.post_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.se = SE_net(32, 32)
        self.final = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.pre_enhance(x)
        enhanced = self.enhance(x)
        x = torch.cat((enhanced, x), dim=1)
        out = self.se(self.post_conv(x))
        out = self.final(out)
        return F.sigmoid(out)


@ARCH_REGISTRY.register()
class AWNet4(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 block=[2, 2, 2, 3, 4]):
        super().__init__()

        self.conv1 = nn.Conv2d(num_in_ch, 64, kernel_size=3, stride=1, padding=1)
        # layer1
        _layer_1_dw = []
        for i in range(block[0]):
            _layer_1_dw.append(GCRDB(64, ContextBlock2d))
        _layer_1_dw.append(GCWTResDown(64, ContextBlock2d, norm_layer=None))
        self.layer1 = nn.Sequential(*_layer_1_dw)

        # layer 2
        _layer_2_dw = []
        for i in range(block[1]):
            _layer_2_dw.append(GCRDB(128, ContextBlock2d))
        _layer_2_dw.append(GCWTResDown(128, ContextBlock2d, norm_layer=None))
        self.layer2 = nn.Sequential(*_layer_2_dw)

        # layer 3
        _layer_3_dw = []
        for i in range(block[2]):
            _layer_3_dw.append(GCRDB(256, ContextBlock2d))
        _layer_3_dw.append(GCWTResDown(256, ContextBlock2d, norm_layer=None))
        self.layer3 = nn.Sequential(*_layer_3_dw)

        # layer 4
        _layer_4_dw = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        _layer_4_dw.append(GCWTResDown(512, ContextBlock2d, norm_layer=None))
        self.layer4 = nn.Sequential(*_layer_4_dw)

        # layer 5
        _layer_5_dw = []
        for i in range(block[4]):
            _layer_5_dw.append(GCRDB(1024, ContextBlock2d))
        self.layer5 = nn.Sequential(*_layer_5_dw)

        # upsample4
        self.layer4_up = GCIWTResUp(2048, ContextBlock2d)

        # upsample3
        self.layer3_up = GCIWTResUp(1024, ContextBlock2d)

        # upsample2
        self.layer2_up = GCIWTResUp(512, ContextBlock2d)

        # upsample1
        self.layer1_up = GCIWTResUp(256, ContextBlock2d)

        self.sc_x1 = shortcutblock(64, 64)
        self.sc_x2 = shortcutblock(128, 128)
        self.sc_x3 = shortcutblock(256, 256)
        self.sc_x4 = shortcutblock(512, 512)

        self.scale_5 = nn.Conv2d(1024, num_out_ch, kernel_size=3, padding=1)
        self.scale_4 = nn.Conv2d(512, num_out_ch, kernel_size=3, padding=1)
        self.scale_3 = nn.Conv2d(256, num_out_ch, kernel_size=3, padding=1)
        self.scale_2 = nn.Conv2d(128, num_out_ch, kernel_size=3, padding=1)
        self.scale_1 = nn.Conv2d(64, num_out_ch, kernel_size=3, padding=1)

        self.se1 = SE_net(64, 64)
        self.se2 = SE_net(128, 128)
        self.se3 = SE_net(256, 256)
        self.se4 = SE_net(512, 512)
        self.se5 = SE_net(1024, 1024)

        self.last = last_upsample()

    def forward(self, x, target=None, teacher_latent=None):

        x1 = self.conv1(x)

        x2, x2_dwt = self.layer1(self.se1(x1))
        x3, x3_dwt = self.layer2(self.se2(x2))
        x4, x4_dwt = self.layer3(self.se3(x3))
        x5, x5_dwt = self.layer4(self.se4(x4))
        x5_latent = self.layer5(self.se5(x5))

        x5_out = self.scale_5(x5_latent)
        x5_out = F.sigmoid(x5_out)
        x4_up = self.layer4_up(x5_latent, x5_dwt) + self.sc_x4(x4)
        x4_out = self.scale_4(x4_up)
        x4_out = F.sigmoid(x4_out)
        x3_up = self.layer3_up(x4_up, x4_dwt) + self.sc_x3(x3)
        x3_out = self.scale_3(x3_up)
        x3_out = F.sigmoid(x3_out)
        x2_up = self.layer2_up(x3_up, x3_dwt) + self.sc_x2(x2)
        x2_out = self.scale_2(x2_up)
        x2_out = F.sigmoid(x2_out)
        x1_up = self.layer1_up(x2_up, x2_dwt) + self.sc_x1(x1)
        x1_out = self.scale_1(x1_up)
        x1_out = F.sigmoid(x1_out)
        out = self.last(x1_up)
        # return (out, x1_out, x2_out, x3_out, x4_out, x5_out), x5_latent
        return out

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = AWNet(in_channels=4, out_channels=3)
    print(count_parameters(net))

    data = torch.randn(1, 4, 224, 224)
    print(net(data).size())