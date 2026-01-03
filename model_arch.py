from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from CAFFM import CAFFM
import torch.nn as nn
import torch
import torch.nn.functional as F


class BlurDetector(nn.Module):
    def __init__(self, out_classes=2, scale_ch=64):
        super().__init__()
        self.backbone1 = resnet152(weights='IMAGENET1K_V1')
        self.backbone2 = resnet50(weights='IMAGENET1K_V1')

        def hook_layers(net):
            return [
                torch.nn.Sequential(
                    net.conv1,
                    net.bn1,
                    net.relu,
                    net.maxpool,
                ),
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4
            ]

        self.hooks1 = hook_layers(self.backbone1)
        self.hooks2 = hook_layers(self.backbone2)

        # self.caffm = CAFFM([64, 64, 128, 256, 512], scale_ch)
        self.caffm = CAFFM([64, 256, 512, 1024, 2048], scale_ch)
        self.upsample_branch = nn.Sequential(
            nn.ConvTranspose2d(scale_ch, scale_ch, 4, stride=2, padding=1),
            nn.Conv2d(scale_ch, scale_ch, 3, padding=1),
            nn.BatchNorm2d(scale_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(scale_ch, scale_ch, 4, stride=2, padding=1),
            nn.Conv2d(scale_ch, scale_ch, 3, padding=1),
            nn.BatchNorm2d(scale_ch), 
            nn.ReLU(inplace=True),
            nn.Conv2d(scale_ch, out_classes, 1, 1, padding=0),
        )
        self.direct_branch = nn.Sequential(
            nn.Conv2d(scale_ch, out_classes, 1, 1, padding=0)
        )

        self.proj_head = nn.Sequential(
            nn.Conv2d(scale_ch, scale_ch, 1, 1, padding=0),
        )

    def forward(self, x, proj=False):
        feats = []
        f = x
        for i, layer in enumerate(self.hooks1):
            f = layer(f)
            feats.append(f)
        feats2 = []
        # print([f.shape for f in feats])
        f = x
        for i, layer in enumerate(self.hooks2):
            f = layer(f)
            feats2.append(f)
        # print([f.shape for f in feats2])
        # exit()
        feats = [torch.cat([feats[i], feats2[i]], dim=1) for i in range(5)]
        f_fused = self.caffm(feats)
        # out = self.head(self.head[0](f_fused))
        # out_up = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_up = self.upsample_branch(f_fused)
        out = self.direct_branch(f_fused)
        if proj:
            proj = self.proj_head(f_fused)
            return out_up, out, proj
        else:
            return out_up, out