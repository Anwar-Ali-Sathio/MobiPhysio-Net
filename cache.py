"""On-disk caches for decoded frames and backbone features."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class FrameCache:
    """``<data_dir>/frames_T{t}_S{size}/<video_id>.npz`` with ``frames`` + ``mask``."""

    def __init__(self, data_dir: Path, t: int, size: int):
        self.root = Path(data_dir) / f"frames_T{t}_S{size}"
        self.status_dir = Path(data_dir) / f"status_T{t}_S{size}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def path(self, video_id: str) -> Path:
        return self.root / f"{video_id}.npz"

    def has(self, video_id: str) -> bool:
        return self.path(video_id).exists()

    def save(self, video_id: str, frames: np.ndarray, mask: np.ndarray) -> None:
        tmp = self.root / f".{video_id}.tmp.npz"
        np.savez(tmp, frames=frames, mask=mask)
        os.replace(tmp, self.path(video_id))

    def load(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.path(video_id)) as z:
            return z["frames"], z["mask"]

    def write_status(self, video_id: str, status: dict) -> None:
        (self.status_dir / f"{video_id}.json").write_text(json.dumps(status))

    def read_status(self, video_id: str) -> dict | None:
        p = self.status_dir / f"{video_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def all_status(self) -> dict[str, dict]:
        return {p.stem: json.loads(p.read_text()) for p in self.status_dir.glob("*.json")}


class FeatureStore:
    """Dense per-backbone feature arrays: ``<data_dir>/features/<name>.npz``.

    Holds ``video_ids`` (N,), ``feats`` (N, T, D) float16 and ``mask`` (N, T).
    """

    def __init__(self, data_dir: Path, experiment: str = "full"):
        self.root = Path(data_dir) / "features" / experiment
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.npz"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, name: str, video_ids, feats, mask, **extra) -> None:
        tmp = self.root / f".{name}.tmp.npz"
        np.savez(tmp, video_ids=np.asarray(video_ids), feats=feats, mask=mask, **extra)
        os.replace(tmp, self.path(name))

    def load(self, name: str) -> dict:
        with np.load(self.path(name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
