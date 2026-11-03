"""CNN-LSTM baseline: from-scratch frame CNN (same trunk as 2D-CNN) followed by an
LSTM, trained jointly end to end on 112x112 RGB frames."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .cnn2d import FrameCNN


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 9, widths=(32, 64, 128, 256), feat_dim: int = 256,
                 hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.cnn = FrameCNN(widths, feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        f = self.cnn(x.flatten(0, 1)).view(b, t, -1)
        lengths = mask.bool().sum(1).clamp_min(1).cpu()
        _, (h_n, _) = self.lstm(pack_padded_sequence(f, lengths, batch_first=True,
                                                     enforce_sorted=False))
        return self.head(h_n[-1])
