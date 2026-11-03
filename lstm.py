"""LSTM baseline over temporally ordered frame features (frozen ImageNet ResNet50
embeddings, 2048-d). Prediction from the hidden state at the last VALID frame."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceLSTM(nn.Module):
    def __init__(self, in_dim: int = 2048, proj_dim: int = 256, hidden: int = 256,
                 layers: int = 1, num_classes: int = 9, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, proj_dim), nn.ReLU())
        self.lstm = nn.LSTM(proj_dim, hidden, layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.bidirectional = bidirectional
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, num_classes))

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.bool().sum(1).clamp_min(1).cpu()
        packed = pack_padded_sequence(self.proj(x.float()), lengths, batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:  # final forward state ++ final backward state (last layer)
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, mask))


def LSTMClassifier(**kw) -> SequenceLSTM:
    kw["bidirectional"] = False
    return SequenceLSTM(**kw)
