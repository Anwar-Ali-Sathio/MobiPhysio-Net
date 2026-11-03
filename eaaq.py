"""Task B - EAAQ score regression (secondary, exploratory).

Target: per-video average of three physiotherapists' EAAQ scores, 0-100 (higher =
more accurate execution), linked ONLY through explicit video names in the
annotation CSV. Same participant-independent split as Task A.

Models (all fitted on TRAIN, hyper-parameters chosen on VAL):
* train_mean                 - constant training mean
* exercise_mean              - training mean per exercise (exercise assumed known)
* exercise_variation_mean    - training mean per (exercise, recording variation);
                               a confound probe: variation codes describe recording
                               conditions (lighting/jitter/occlusion), not movement
* ridge_vit                  - ridge regression on mean-pooled frozen ViT-S/16 features
* temporal_vit               - MobiPhysio-Net temporal encoder + linear head (MSE)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..evaluation.metrics import regression_metrics


def icc2_1(x: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater. x: (n, k)."""
    x = x[np.isfinite(x).all(1)]
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def _group_mean(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], fallback: float) -> np.ndarray:
    m = train.groupby(cols).y.mean()
    keys = [tuple(r) if len(cols) > 1 else r[0] for r in test[cols].values]
    return np.array([m.get(k, fallback) for k in keys])


def run_task_b(split_df: pd.DataFrame, eaaq: pd.DataFrame, vit: dict, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    df = split_df.merge(eaaq[eaaq.video_name != ""].drop_duplicates("video_name", keep=False)
                        [["video_name", "avg_score", "judge1_score", "judge2_score", "judge3_score"]],
                        left_on="video_id", right_on="video_name", how="inner")
    df = df[np.isfinite(df.avg_score)].sort_values("video_id").reset_index(drop=True)
    df["y"] = df.avg_score.astype(float)
    pos = {v: i for i, v in enumerate(vit["video_ids"])}
    sel = np.array([pos[v] for v in df.video_id])
    Z = vit["feats"][sel].astype(np.float32)
    M = vit["mask"][sel]
    pooled = (Z * M[..., None]).sum(1) / M.sum(1, keepdims=True).clip(1)
    tr, va, te = (np.where(df.split == s)[0] for s in ["train", "val", "test"])
    y = df.y.values
    preds = {}
    mu = y[tr].mean()
    preds["train_mean"] = np.full(len(te), mu)
    preds["exercise_mean"] = _group_mean(df.iloc[tr], df.iloc[te], ["exercise"], mu)
    preds["exercise_variation_mean"] = _group_mean(df.iloc[tr], df.iloc[te], ["exercise", "variation"], mu)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(pooled[tr])
    P = sc.transform(pooled)
    grid = []
    for a in [1, 10, 100, 1000, 10000]:
        r = Ridge(alpha=a).fit(P[tr], y[tr])
        grid.append((float(np.abs(r.predict(P[va]) - y[va]).mean()), a))
    best_a = min(grid)[1]
    preds["ridge_vit"] = Ridge(alpha=best_a).fit(P[tr], y[tr]).predict(P[te])

    from ..models.mobiphysio_net import TemporalEncoder
    enc = TemporalEncoder(Z.shape[-1], 128, 2, 4, 256, 0.2)
    head = nn.Linear(128, 1)
    net = nn.ModuleDict({"enc": enc, "head": head})
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)
    Zt, Mt = torch.from_numpy(Z), torch.from_numpy(M)
    yt = torch.from_numpy(((y - mu) / y[tr].std()).astype(np.float32))
    rng = np.random.default_rng(seed)

    def fwd(idx):
        _, g = enc(Zt[idx], Mt[idx])
        return head(g).squeeze(1)

    best, best_mae, hist, bad = None, np.inf, [], 0
    for ep in range(60):
        net.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), 32):
            b = perm[s:s + 32]
            loss = nn.functional.mse_loss(fwd(b), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = fwd(va).numpy() * y[tr].std() + mu
        mae = float(np.abs(pv - y[va]).mean())
        hist.append({"epoch": ep + 1, "val_mae": mae})
        if mae < best_mae:
            best_mae, best, bad = mae, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(best)
    net.eval()
    with torch.no_grad():
        preds["temporal_vit"] = fwd(te).numpy() * y[tr].std() + mu

    rows = []
    for k, p in preds.items():
        rows.append({"model": k, **regression_metrics(y[te], p)})
    judges = df[["judge1_score", "judge2_score", "judge3_score"]].values.astype(float)
    var_means = df.groupby("variation").y.agg(["mean", "std", "count"]).reset_index()
    pred_df = df.iloc[te][["video_id", "participant", "exercise", "variation", "angle", "y"]].copy()
    for k, p in preds.items():
        pred_df[f"pred_{k}"] = p
    return {"metrics": pd.DataFrame(rows), "predictions": pred_df,
            "n": {"linked_usable": int(len(df)), "train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "ridge_alpha": best_a, "temporal_history": hist,
            "inter_judge_icc2_1": icc2_1(judges), "mean_judge_sd": float(np.nanstd(judges, axis=1).mean()),
            "score_by_variation": var_means,
            "score_summary": {"mean": float(y.mean()), "sd": float(y.std()), "min": float(y.min()),
                              "max": float(y.max())}}
