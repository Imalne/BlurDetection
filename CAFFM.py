import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return x * w

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class CAFFM(nn.Module):
    def __init__(self, channels_per_scale, output_channel):
        super().__init__()
        self.up1 = UpsampleBlock(channels_per_scale[0]*2, output_channel)
        self.up2 = UpsampleBlock(channels_per_scale[1]*2, output_channel)
        self.up3 = UpsampleBlock(channels_per_scale[2]*2, output_channel)
        self.up4 = UpsampleBlock(channels_per_scale[3]*2, output_channel)
        self.up5 = UpsampleBlock(channels_per_scale[4]*2, output_channel)
        self.attn = ChannelAttention(5 * output_channel)
        self.fuse_conv = nn.Conv2d(5*output_channel, output_channel, 1)

    def forward(self, feats):
        H, W = feats[0].shape[2], feats[0].shape[3]
        size = (H, W)
        f1 = self.up1(feats[0], size)
        f2 = self.up2(feats[1], size)
        f3 = self.up3(feats[2], size)
        f4 = self.up4(feats[3], size)
        f5 = self.up5(feats[4], size)
        Fcat = torch.cat([f1, f2, f3, f4, f5], dim=1)
        Fatt = self.attn(Fcat)
        Fout = self.fuse_conv(Fatt)
        return Fout
