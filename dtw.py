"""Algorithm 5 - CONSTRAINED_DTW (reference NumPy + batched PyTorch implementations).

Conventions (identical in both implementations):

* local cost      c(i, j) = || h_i - r_j ||_2
* recursion       D(i, j) = c(i, j) + min(D(i-1, j-1), D(i-1, j), D(i, j-1)),
                  D(0, 0) = 0, D(i, 0) = D(0, j) = +inf
* tie-break       diagonal, then (i-1, j), then (i, j-1)
* path length     P(i, j) = P(predecessor) + 1, so P(T, L) = |pi| of the selected path
* normalisation   d = D(T, L) / |pi|
* constraint      optional Sakoe-Chiba band on the length-normalised diagonal:
                  cell (i, j) allowed iff | i - j * T / L | <= w, with
                  w = max(band, |T - L|) so the end cell is always reachable.
* masks           padding is at the END of a sequence (see preprocessing); only the
                  valid prefix (length = mask.sum()) participates.

Standard DTW is NOT differentiable through the alignment; in MobiPhysio-Net it is
used as a no-gradient distance-feature branch.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class DTWError(ValueError):
    pass


def _band_width(t: int, u: int, band: int | None) -> float:
    if band is None:
        return math.inf
    return float(max(band, abs(t - u)))


def dtw_numpy(h: np.ndarray, r: np.ndarray, band: int | None = None,
              h_mask: np.ndarray | None = None, r_mask: np.ndarray | None = None,
              return_path: bool = False):
    """Reference DTW between ``h`` (T, d) and ``r`` (L, d).

    Returns ``d`` or ``(d, path, D)`` with ``path`` a list of 0-based (i, j) pairs.
    """
    h = np.asarray(h, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if h.ndim != 2 or r.ndim != 2 or h.shape[1] != r.shape[1]:
        raise DTWError(f"expected (T,d) and (L,d) with equal d, got {h.shape}, {r.shape}")
    if h_mask is not None:
        h = _valid_prefix(h, h_mask)
    if r_mask is not None:
        r = _valid_prefix(r, r_mask)
    if len(h) == 0 or len(r) == 0:
        raise DTWError("empty sequence after masking")
    if not (np.isfinite(h).all() and np.isfinite(r).all()):
        raise DTWError("non-finite values in DTW input")
    T, L = len(h), len(r)
    w = _band_width(T, L, band)
    cost = np.sqrt(np.maximum(((h[:, None, :] - r[None, :, :]) ** 2).sum(-1), 0.0))
    D = np.full((T + 1, L + 1), np.inf)
    P = np.zeros((T + 1, L + 1), dtype=np.int64)
    B = np.zeros((T + 1, L + 1), dtype=np.int8)  # 0 diag, 1 up (i-1,j), 2 left (i,j-1)
    D[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, L + 1):
            if abs(i - j * T / L) > w:
                continue
            cands = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = 0 if cands[0] <= min(cands[1], cands[2]) else (1 if cands[1] <= cands[2] else 2)
            if not np.isfinite(cands[k]):
                continue
            D[i, j] = cost[i - 1, j - 1] + cands[k]
            pi, pj = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            P[i, j] = P[pi, pj] + 1
            B[i, j] = k
    if not np.isfinite(D[T, L]):
        raise DTWError("end cell unreachable under the warping constraint")
    d = D[T, L] / P[T, L]
    if not return_path:
        return float(d)
    path, i, j = [], T, L
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = B[i, j]
        i, j = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
    path.reverse()
    assert len(path) == P[T, L]
    return float(d), path, D


def _valid_prefix(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != x.shape[0]:
        raise DTWError(f"mask length {mask.shape[0]} != sequence length {x.shape[0]}")
    n = int(mask.sum())
    if not mask[:n].all():
        raise DTWError("mask must be a contiguous valid prefix (padding at the end)")
    return x[:n]


@torch.no_grad()
def dtw_batch(q: torch.Tensor, q_mask: torch.Tensor, r: torch.Tensor, r_mask: torch.Tensor,
              band: int | None = None) -> torch.Tensor:
    """All-pairs DTW: queries ``q`` (B, T, d) vs references ``r`` (R, L, d).

    Returns (B, R) path-normalised distances. No gradient flows through this op.
    """
    if q.ndim != 3 or r.ndim != 3 or q.shape[-1] != r.shape[-1]:
        raise DTWError(f"bad shapes {tuple(q.shape)} vs {tuple(r.shape)}")
    if not (torch.isfinite(q).all() and torch.isfinite(r).all()):
        raise DTWError("non-finite values in DTW input")
    q, r = q.detach().float(), r.detach().float()
    B, T, _ = q.shape
    R, L, _ = r.shape
    lq = q_mask.bool().sum(1)  # (B,)
    lr = r_mask.bool().sum(1)  # (R,)
    if (lq == 0).any() or (lr == 0).any():
        raise DTWError("empty sequence after masking")
    cost = torch.cdist(q.reshape(B * T, -1), r.reshape(R * L, -1)).reshape(B, T, R, L)
    cost = cost.permute(0, 2, 1, 3)  # (B, R, T, L)
    # Band allowance per pair, using that pair's valid lengths.
    ii = torch.arange(1, T + 1, dtype=torch.float32).view(1, 1, T, 1)
    jj = torch.arange(1, L + 1, dtype=torch.float32).view(1, 1, 1, L)
    Tq = lq.float().view(B, 1, 1, 1)
    Lr = lr.float().view(1, R, 1, 1)
    if band is None:
        allowed = torch.ones(B, R, T, L, dtype=torch.bool)
    else:
        w = torch.maximum(torch.full_like(Tq - Lr, float(band)), (Tq - Lr).abs())
        allowed = (ii - jj * Tq / Lr).abs() <= w
    inf = torch.tensor(float("inf"))
    D = torch.full((B, R, T + 1, L + 1), float("inf"))
    P = torch.zeros((B, R, T + 1, L + 1))
    D[:, :, 0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, L + 1):
            diag, up, left = D[:, :, i - 1, j - 1], D[:, :, i - 1, j], D[:, :, i, j - 1]
            take_diag = diag <= torch.minimum(up, left)
            take_up = (~take_diag) & (up <= left)
            best = torch.where(take_diag, diag, torch.where(take_up, up, left))
            plen = torch.where(take_diag, P[:, :, i - 1, j - 1],
                               torch.where(take_up, P[:, :, i - 1, j], P[:, :, i, j - 1]))
            ok = allowed[:, :, i - 1, j - 1] & torch.isfinite(best)
            D[:, :, i, j] = torch.where(ok, cost[:, :, i - 1, j - 1] + best, inf)
            P[:, :, i, j] = torch.where(ok, plen + 1, torch.zeros_like(plen))
    bi = torch.arange(B).view(B, 1).expand(B, R)
    ri = torch.arange(R).view(1, R).expand(B, R)
    end_i = lq.view(B, 1).expand(B, R)
    end_j = lr.view(1, R).expand(B, R)
    Dend = D[bi, ri, end_i, end_j]
    Pend = P[bi, ri, end_i, end_j]
    if not torch.isfinite(Dend).all():
        raise DTWError("end cell unreachable under the warping constraint")
    return Dend / Pend
