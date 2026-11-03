"""SVM baseline on fixed-length handcrafted video descriptors.

Descriptor (per video, from the cached 16 letterboxed frames):
* appearance: HOG (64x64 grayscale, 8x8 cells, 9 bins) per valid frame ->
  temporal mean and std;
* motion: absolute frame differences of consecutive valid frames on a 16x16
  grayscale grid -> mean, std and max over time (motion-energy image statistics),
  plus a 16-bin per-step global motion-magnitude profile.
Scaling is fitted on TRAIN only; C/gamma are selected by validation macro F1.
"""
from __future__ import annotations

import cv2
import numpy as np


def hog(gray: np.ndarray, cell: int = 8, bins: int = 9, block: int = 2) -> np.ndarray:
    """Dalal-Triggs HOG (unsigned gradients, 2x2-cell blocks, stride 1 cell, L2-Hys).

    Implemented in NumPy because OpenCV >= 5 no longer ships ``cv2.HOGDescriptor``.
    """
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    b = ang / (180.0 / bins)
    b0 = np.floor(b).astype(int) % bins
    w1 = b - np.floor(b)
    b1 = (b0 + 1) % bins
    ch, cw = g.shape[0] // cell, g.shape[1] // cell
    hist = np.zeros((ch, cw, bins), np.float32)
    cy = (np.arange(ch * cell) // cell)[:, None].repeat(cw * cell, 1)
    cx = (np.arange(cw * cell) // cell)[None, :].repeat(ch * cell, 0)
    sl = (slice(0, ch * cell), slice(0, cw * cell))
    np.add.at(hist, (cy, cx, b0[sl]), (mag * (1 - w1))[sl])
    np.add.at(hist, (cy, cx, b1[sl]), (mag * w1)[sl])
    feats = []
    for i in range(ch - block + 1):
        for j in range(cw - block + 1):
            v = hist[i:i + block, j:j + block].ravel()
            v = v / np.sqrt((v ** 2).sum() + 1e-6)
            v = np.minimum(v, 0.2)
            feats.append(v / np.sqrt((v ** 2).sum() + 1e-6))
    return np.concatenate(feats)


def video_descriptor(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = frames[mask.astype(bool)]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in v]
    hogs = np.stack([hog(cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA))
                     for g in gray])
    small = np.stack([cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                      for g in gray])
    if len(small) > 1:
        diff = np.abs(np.diff(small, axis=0))
    else:
        diff = np.zeros((1, 16, 16), np.float32)
    prof = diff.reshape(len(diff), -1).mean(1)
    prof16 = np.interp(np.linspace(0, len(prof) - 1, 16), np.arange(len(prof)), prof)
    return np.concatenate([hogs.mean(0), hogs.std(0), diff.mean(0).ravel(), diff.std(0).ravel(),
                           diff.max(0).ravel(), prof16]).astype(np.float32)
