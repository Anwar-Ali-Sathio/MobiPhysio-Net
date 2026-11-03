"""ResNet50 baseline: ImageNet-pretrained torchvision ResNet50 (IMAGENET1K_V2),
backbone FROZEN (CPU budget), trainable 9-class head on the 2048-d pooled frame
features; video logits = masked mean of frame logits."""
from __future__ import annotations

import torch
from torch import nn

from .mobiphysio_net import masked_mean


def resnet50_backbone(pretrained: bool = True) -> nn.Module:
    import torchvision
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    m.fc = nn.Identity()
    return m.eval()


class ResNet50Head(nn.Module):
    def __init__(self, in_dim: int = 2048, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x (B,T,2048)
        return masked_mean(self.head(x.float()), mask)
