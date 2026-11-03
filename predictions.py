"""Prediction file format: one row per test video."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["video_id", "participant", "y_true", "y_pred"]


def save_predictions(path: Path, video_ids, participants, y_true, y_pred, proba,
                     class_codes: list[str], dtw=None, extra: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"video_id": video_ids, "participant": participants,
                       "y_true": np.asarray(y_true, int), "y_pred": np.asarray(y_pred, int)})
    df["true_exercise"] = [class_codes[i] for i in df.y_true]
    df["pred_exercise"] = [class_codes[i] for i in df.y_pred]
    if proba is not None:
        for k, c in enumerate(class_codes):
            df[f"p_{c}"] = np.asarray(proba)[:, k]
    if dtw is not None:
        for k, c in enumerate(class_codes):
            df[f"dtw_{c}"] = np.asarray(dtw)[:, k]
    for k, v in (extra or {}).items():
        df[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_predictions(path: Path, class_codes: list[str]):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    pcols = [f"p_{c}" for c in class_codes]
    proba = df[pcols].values if all(c in df for c in pcols) else None
    return df, proba
