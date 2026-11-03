"""2D-CNN baseline: frame-level CNN trained from scratch; video logits = masked mean
of per-frame logits (documented video-level aggregation)."""
from __future__ import annotations

import torch
from torch import nn

from .mobiphysio_net import masked_mean


def conv_bn(cin: int, cout: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)]


class FrameCNN(nn.Module):
    """Compact CNN trunk for 112x112 RGB frames -> ``out_dim`` vector.

    Stride-2 stem (-> 56x56), then one conv + 2x2 max-pool per further stage
    (56 -> 28 -> 14 -> 7) and global average pooling: ~0.18 GFLOP per frame,
    sized for CPU-only training (a two-conv VGG-style variant was ~4x slower).
    """

    def __init__(self, widths=(32, 64, 128, 256), out_dim: int = 256):
        super().__init__()
        layers = conv_bn(3, widths[0], stride=2)
        cin = widths[0]
        for w in widths[1:]:
            layers += conv_bn(cin, w) + [nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,3,H,W)
        return torch.relu(self.fc(self.pool(self.features(x)).flatten(1)))


class CNN2D(nn.Module):
    def __init__(self, num_classes: int = 9, widths=(32, 64, 128, 256), feat_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.cnn = FrameCNN(widths, feat_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x (B,T,3,H,W)
        b, t = x.shape[:2]
        frame_logits = self.head(self.cnn(x.flatten(0, 1))).view(b, t, -1)
        return masked_mean(frame_logits, mask)
