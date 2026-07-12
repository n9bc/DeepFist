"""Pure-CNN + CTC network: 2D freq-collapsing stem -> 1D dilated TCN -> logits."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfist.morse.alphabet import TOKENS

N_CLASSES = len(TOKENS)


class ConvBN2d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResTCN(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        return F.relu(x + self.bn(self.conv(x)), inplace=True)


def _ch(base: int, width: float) -> int:
    """Scale a base channel count by `width`, rounded to a multiple of 8 (>=8)."""
    return max(8, int(round(base * width / 8)) * 8)


class CwCtcNet(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, time_downsample: int = 2,
                 width: float = 1.0):
        super().__init__()
        assert time_downsample in (2, 4)
        s4_t = 2 if time_downsample == 4 else 1
        c1, c2, c3, c4 = (_ch(32, width), _ch(48, width), _ch(64, width), _ch(96, width))
        t = _ch(128, width)
        # Stem stride is (freq, time). S3 always halves time; S4 halves again only for 4x.
        self.stem = nn.ModuleList([
            ConvBN2d(1, c1, stride=(1, 1)),
            ConvBN2d(c1, c2, stride=(2, 1)),
            ConvBN2d(c2, c3, stride=(2, 2)),
            ConvBN2d(c3, c4, stride=(2, s4_t)),
        ])
        self.proj = nn.Conv1d(c4, t, 1, bias=False)
        self.proj_bn = nn.BatchNorm1d(t)
        self.tcn = nn.ModuleList([ResTCN(t, d) for d in (1, 2, 4, 8, 16, 32, 64, 1)])
        self.head = nn.Conv1d(t, t, 1, bias=False)
        self.head_bn = nn.BatchNorm1d(t)
        self.classifier = nn.Conv1d(t, n_classes, 1)

    def forward(self, x):
        for layer in self.stem:
            x = layer(x)
        x = torch.amax(x, dim=2)                       # max-pool over freq -> [B,96,T']
        x = F.relu(self.proj_bn(self.proj(x)), inplace=True)
        for block in self.tcn:
            x = block(x)
        x = F.relu(self.head_bn(self.head(x)), inplace=True)
        x = self.classifier(x)                         # [B,C,T']
        x = x.permute(2, 0, 1)                          # [T',B,C]
        return F.log_softmax(x.float(), dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
