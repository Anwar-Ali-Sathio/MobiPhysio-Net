"""Algorithm 4 - BUILD_REFERENCE_BANK (training-only DTW k-medoids per class)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from .dtw import dtw_batch


class LeakageError(RuntimeError):
    pass


def _pairwise_dtw(x: torch.Tensor, m: torch.Tensor, band: int | None, chunk: int = 64) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros((n, n))
    for s in range(0, n, chunk):
        out[s:s + chunk] = dtw_batch(x[s:s + chunk], m[s:s + chunk], x, m, band).numpy()
    return out


def k_medoids(dist: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50) -> list[int]:
    """Simple alternating k-medoids on a precomputed distance matrix.

    Initialised deterministically with the global medoid, then farthest-point
    additions; k=1 returns the DTW medoid (argmin of summed distances).
    """
    n = dist.shape[0]
    k = min(k, n)
    med = [int(np.argmin(dist.sum(1)))]
    while len(med) < k:
        med.append(int(np.argmax(dist[:, med].min(1))))
    for _ in range(iters):
        assign = np.argmin(dist[:, med], axis=1)
        new = []
        for c in range(k):
            members = np.where(assign == c)[0]
            if len(members) == 0:
                new.append(med[c])
                continue
            sub = dist[np.ix_(members, members)]
            new.append(int(members[np.argmin(sub.sum(1))]))
        if new == med:
            break
        med = new
    return med


def build_reference_bank(feats: np.ndarray, masks: np.ndarray, labels: np.ndarray,
                         video_ids: np.ndarray, train_ids: set[str], heldout_ids: set[str],
                         num_classes: int, refs_per_class: int = 3, sample_limit: int = 200,
                         band: int | None = None, seed: int = 0) -> dict:
    """Select ``refs_per_class`` DTW k-medoids per class from TRAINING videos only.

    ``feats`` are the sequences in which medoids are selected (frozen ViT
    features, L2-normalised by the caller). For tractability at most
    ``sample_limit`` training videos per class enter the distance matrix (random,
    seeded). Raises ``LeakageError`` if any selected reference is held out.
    """
    rng = np.random.default_rng(seed)
    is_train = np.array([v in train_ids for v in video_ids])
    if not is_train.any():
        raise LeakageError("no training videos supplied")
    bank = {"refs_per_class": refs_per_class, "sample_limit": sample_limit, "band": band,
            "seed": seed, "selection_space": "frozen ViT CLS features (L2-normalised)",
            "classes": {}}
    for k in range(num_classes):
        idx = np.where(is_train & (labels == k))[0]
        if len(idx) == 0:
            raise LeakageError(f"class {k} has no training examples - cannot build a reference")
        if len(idx) > sample_limit:
            idx = np.sort(rng.choice(idx, sample_limit, replace=False))
        x = torch.from_numpy(np.asarray(feats[idx], dtype=np.float32))
        m = torch.from_numpy(np.asarray(masks[idx], dtype=bool))
        dist = _pairwise_dtw(x, m, band)
        med = k_medoids(dist, refs_per_class, rng)
        bank["classes"][str(k)] = {
            "video_ids": [str(video_ids[idx[i]]) for i in med],
            "n_candidates": int(len(idx)),
            "mean_within_class_dtw": float(dist[np.triu_indices(len(idx), 1)].mean()) if len(idx) > 1 else 0.0,
        }
    verify_reference_provenance(bank, train_ids, heldout_ids)
    bank["training_ids_sha256"] = hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
    bank["reference_ids_sha256"] = hashlib.sha256(
        json.dumps(reference_ids(bank)).encode()).hexdigest()
    return bank


def reference_ids(bank: dict) -> list[str]:
    return [v for k in sorted(bank["classes"], key=int) for v in bank["classes"][k]["video_ids"]]


def reference_labels(bank: dict) -> list[int]:
    return [int(k) for k in sorted(bank["classes"], key=int) for _ in bank["classes"][k]["video_ids"]]


def verify_reference_provenance(bank: dict, train_ids: set[str], heldout_ids: set[str]) -> None:
    ids = reference_ids(bank)
    bad = [v for v in ids if v in heldout_ids or v not in train_ids]
    if bad:
        raise LeakageError(f"reference bank contains non-training videos: {bad}")


def aggregate_class_distances(d: torch.Tensor, ref_labels: list[int], num_classes: int,
                              rule: str = "min") -> torch.Tensor:
    """(B, R) query-to-reference distances -> (B, num_classes) class distances."""
    lab = torch.tensor(ref_labels)
    out = []
    for k in range(num_classes):
        dk = d[:, lab == k]
        out.append(dk.min(1).values if rule == "min" else dk.mean(1))
    return torch.stack(out, 1)
