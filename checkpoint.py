"""Checkpoint save/load with SHA-256 hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_checkpoint(path: Path, model: torch.nn.Module, meta: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return sha256_file(path)


def load_checkpoint(path: Path, model: torch.nn.Module) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    return ck["meta"]
