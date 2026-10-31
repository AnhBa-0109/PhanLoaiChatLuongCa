import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNReLU, ResidualBlock

class MaskedGAP(nn.Module):
    def forward(self, feat, mask):
        m = F.interpolate(mask, size=feat.shape[-2:], mode="area")
        num = (feat * m).sum(dim=(2, 3))            
        den = m.sum(dim=(2, 3)).clamp_min(1e-6)     
        return num / den

def make_stage(in_ch, out_ch, n_blocks, stride, use_se=True):
    for _ in range(n_blocks - 1):
        layers.append(ResidualBlock(out_ch, out_ch, 1, use_se))
    return nn.Sequential(*layers)

class TextureBranch(nn.Module):
    def __init__(self, in_ch=4, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), use_se=True):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, widths[0], 3, stride=2)
        stages, prev = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2
            stages.append(make_stage(prev, w, n, stride, use_se))
            prev = w
        self.stages = nn.Sequential(*stages)
        self.pool = MaskedGAP()
        self.out_dim = widths[-1]

    def forward(self, x):
        mask = x[:, 3:4]         
        feat = self.stages(self.stem(x))
        return self.pool(feat, mask)

    def feature_map(self, x):
        return self.stages(self.stem(x))