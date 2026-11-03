"""Participant-independent splitting and leakage audits."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def grouped_stratified_split(df: pd.DataFrame, fractions: dict, num_classes: int,
                             trials: int = 2000, seed: int = 42,
                             group_col: str = "participant", label_col: str = "label") -> dict:
    """Search random participant partitions; keep the most class-balanced valid one.

    Score = max over (split, class) of |p_split(class) - p_global(class)| +
            max over splits of |video_fraction - target|. Labels only - no model.
    Every class must be present in every split.
    """
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df[group_col].unique()))
    names = list(fractions)
    target = np.array([fractions[n] for n in names])
    counts = df.groupby([group_col, label_col]).size().unstack(fill_value=0)
    counts = counts.reindex(index=groups, columns=range(num_classes), fill_value=0).values
    glob = counts.sum(0) / counts.sum()
    best, best_score = None, np.inf
    for _ in range(trials):
        perm = rng.permutation(len(groups))
        n_g = np.round(target * len(groups)).astype(int)
        n_g[0] = len(groups) - n_g[1:].sum()
        assign = np.zeros(len(groups), int)
        start = 0
        for k, n in enumerate(n_g):
            assign[perm[start:start + n]] = k
            start += n
        split_counts = np.stack([counts[assign == k].sum(0) for k in range(len(names))])
        if (split_counts == 0).any():
            continue
        props = split_counts / split_counts.sum(1, keepdims=True)
        vid_frac = split_counts.sum(1) / split_counts.sum()
        score = np.abs(props - glob).max() + np.abs(vid_frac - target).max()
        if score < best_score:
            best_score, best = score, assign.copy()
    if best is None:
        raise RuntimeError("no participant partition puts every class in every split")
    mapping = {g: names[k] for g, k in zip(groups, best)}
    return {"participant_to_split": mapping, "score": float(best_score), "trials": trials,
            "seed": seed}


def audit_split(df: pd.DataFrame, split_col: str = "split") -> dict:
    parts = {s: set(df.loc[df[split_col] == s, "participant"]) for s in df[split_col].unique()}
    overlap = {f"{a}&{b}": sorted(parts[a] & parts[b]) for a in parts for b in parts if a < b}
    md5 = df.groupby("md5")[split_col].nunique()
    cross_dup = md5[md5 > 1].index.tolist()
    ids = df.groupby("video_id")[split_col].nunique()
    return {"participant_overlap": overlap,
            "participant_overlap_any": any(len(v) for v in overlap.values()),
            "cross_split_duplicate_md5": cross_dup,
            "video_in_multiple_splits": ids[ids > 1].index.tolist(),
            "counts": df.groupby(split_col).size().to_dict(),
            "participants": {k: sorted(v) for k, v in parts.items()}}


def near_duplicate_audit(feats: np.ndarray, splits: np.ndarray, video_ids: np.ndarray,
                         threshold: float = 0.98) -> dict:
    """Nearest cross-split neighbour by cosine of mean-pooled frame embeddings."""
    x = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
    out = {"threshold": threshold, "pairs_above_threshold": [], "max_cross_split_cosine": {},
           "n_pairs_above_threshold": 0}
    names = sorted(set(splits))
    for a in names:
        for b in names:
            if a >= b:
                continue
            ia, ib = np.where(splits == a)[0], np.where(splits == b)[0]
            sim = x[ia] @ x[ib].T
            out["max_cross_split_cosine"][f"{a}-{b}"] = float(sim.max())
            hits = np.argwhere(sim >= threshold)
            out["n_pairs_above_threshold"] += int(len(hits))
            for i, j in hits[:2000]:
                out["pairs_above_threshold"].append(
                    {"a": str(video_ids[ia[i]]), "b": str(video_ids[ib[j]]), "cos": float(sim[i, j])})
    return out


def split_hash(df: pd.DataFrame) -> str:
    s = "\n".join(f"{v},{sp}" for v, sp in sorted(zip(df.video_id, df.split)))
    return hashlib.sha256(s.encode()).hexdigest()
