"""MobiPhysio-Net (Components B-E; Algorithms 2, 3 and 6).

Video -> ViT spatial encoder (shared over frames) -> temporal Transformer encoder
-> DTW distances to a training-only reference bank (no-gradient branch)
-> fusion [g || q] -> MLP -> softmax over 9 exercises.

Spatial encoder modes (``vit_mode``):

* ``frozen``      - ViT-S/16 is frozen; per-frame CLS features are precomputed once
                    (mathematically identical to running the frozen ViT online).
* ``last_block``  - the final ViT transformer block + final LayerNorm are trainable;
                    inputs are the cached token sequences entering that block, so
                    gradients really update ViT weights (partial fine-tuning; full
                    fine-tuning of all 12 blocks is infeasible on the available CPU).

Ablation switches: ``use_dtw`` (False -> "ViT + temporal encoder without DTW"),
``use_temporal`` (False -> "ViT + DTW features without learned temporal encoder":
DTW runs on the L2-normalised ViT features and g is their masked mean).
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .dtw import dtw_batch
from .references import aggregate_class_distances


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """g = sum_t m_t h_t / max(1, sum_t m_t)."""
    m = mask.float().unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


def sinusoidal_positions(t: int, d: int) -> torch.Tensor:
    pos = torch.arange(t).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(t, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class TemporalEncoder(nn.Module):
    """Algorithm 3: projection -> + positional encoding -> masked Transformer."""

    def __init__(self, in_dim: int, d_h: int = 128, layers: int = 2, heads: int = 4,
                 ff: int = 256, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_h), nn.LayerNorm(d_h))
        self.register_buffer("pe", sinusoidal_positions(max_len, d_h), persistent=False)
        layer = nn.TransformerEncoderLayer(d_h, heads, ff, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(z) + self.pe[: z.shape[1]].unsqueeze(0)
        h = self.norm(self.encoder(x, src_key_padding_mask=~mask.bool()))
        h = h * mask.unsqueeze(-1).float()  # padded positions zeroed (never used downstream)
        return h, masked_mean(h, mask)


class LastBlockViT(nn.Module):
    """Trainable final ViT block + norm; input = cached tokens entering that block."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.block = copy.deepcopy(vit.blocks[-1])
        self.norm = copy.deepcopy(vit.norm)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, N_tok, D) -> CLS features (B, T, D)
        b, t, n, d = tokens.shape
        x = self.norm(self.block(tokens.reshape(b * t, n, d).float()))
        return x[:, 0].reshape(b, t, d)


class DistanceScaler(nn.Module):
    """Standardisation of DTW distances; statistics fitted on TRAINING data only."""

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, q: torch.Tensor) -> None:
        self.mean.copy_(q.mean(0))
        self.std.copy_(q.std(0).clamp_min(1e-6))
        self.fitted.fill_(True)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("DistanceScaler used before fitting on training data")
        return (q - self.mean) / self.std


class MobiPhysioNet(nn.Module):
    def __init__(self, vit_dim: int = 384, num_classes: int = 9, d_h: int = 128,
                 layers: int = 2, heads: int = 4, ff: int = 256, dropout: float = 0.1,
                 head_hidden: int = 128, use_dtw: bool = True, use_temporal: bool = True,
                 dtw_band: int | None = None, dtw_aggregation: str = "min",
                 spatial: nn.Module | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.use_dtw, self.use_temporal = use_dtw, use_temporal
        self.dtw_band, self.dtw_aggregation = dtw_band, dtw_aggregation
        self.spatial = spatial  # None -> inputs are precomputed (frozen) ViT features
        self.temporal = TemporalEncoder(vit_dim, d_h, layers, heads, ff, dropout) if use_temporal else None
        g_dim = d_h if use_temporal else vit_dim
        self.scaler = DistanceScaler(num_classes)
        in_dim = g_dim + (num_classes if use_dtw else 0)
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                  nn.Linear(in_dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, num_classes))
        # Reference bank tensors (set by the trainer; inputs in the spatial space).
        self.register_buffer("ref_inputs", torch.zeros(0), persistent=True)
        self.register_buffer("ref_mask", torch.zeros(0, dtype=torch.bool), persistent=True)
        self.ref_labels: list[int] = []
        self._ref_h_cache: torch.Tensor | None = None

    # --- Algorithm 2 + 3 -------------------------------------------------------------
    def spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x) if self.spatial is not None else x.float()

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.spatial_features(x)
        if self.use_temporal:
            return self.temporal(z, mask)
        zn = nn.functional.normalize(z, dim=-1) * mask.unsqueeze(-1).float()
        return zn, masked_mean(z, mask)

    # --- reference bank ----------------------------------------------------------------
    def set_references(self, ref_inputs: torch.Tensor, ref_mask: torch.Tensor, ref_labels: list[int]):
        self.ref_inputs = ref_inputs.float()
        self.ref_mask = ref_mask.bool()
        self.ref_labels = list(ref_labels)
        self._ref_h_cache = None

    @torch.no_grad()
    def refresh_reference_encodings(self) -> torch.Tensor:
        """Encode references with the CURRENT encoder (eval mode, no gradient)."""
        was = self.training
        self.eval()
        h, _ = self.encode(self.ref_inputs, self.ref_mask)
        self.train(was)
        self._ref_h_cache = h
        return h

    # --- Algorithm 6 -------------------------------------------------------------------
    @torch.no_grad()
    def dtw_distances(self, h: torch.Tensor, mask: torch.Tensor,
                      exclude: torch.Tensor | None = None) -> torch.Tensor:
        """(B, num_classes) raw class distances. ``exclude`` (B, R) masks self-matches."""
        if self._ref_h_cache is None:
            self.refresh_reference_encodings()
        d = dtw_batch(h.detach(), mask, self._ref_h_cache, self.ref_mask, self.dtw_band)
        if exclude is not None:
            d = d.masked_fill(exclude, float("inf"))
        return aggregate_class_distances(d, self.ref_labels, self.num_classes, self.dtw_aggregation)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, exclude: torch.Tensor | None = None,
                return_all: bool = False):
        h, g = self.encode(x, mask)
        q_raw = None
        if self.use_dtw:
            q_raw = self.dtw_distances(h, mask, exclude)
            u = torch.cat([g, self.scaler(q_raw)], dim=1)
        else:
            u = g
        logits = self.head(u)
        if return_all:
            return {"logits": logits, "h": h, "g": g, "q": q_raw}
        return logits
