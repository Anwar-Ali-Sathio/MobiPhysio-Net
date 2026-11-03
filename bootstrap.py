"""Participant-level (cluster) bootstrap confidence intervals.

Test participants are resampled with replacement; all videos of a sampled
participant enter the replicate (so within-participant correlation - e.g. the
three simultaneous camera angles of a repetition - is respected). With ~12 test
participants the intervals are wide and themselves approximate.
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

METRICS = {
    "accuracy": lambda y, p, k: skm.accuracy_score(y, p),
    "balanced_accuracy": lambda y, p, k: skm.balanced_accuracy_score(y, p),
    "macro_f1": lambda y, p, k: skm.f1_score(y, p, labels=list(range(k)), average="macro",
                                             zero_division=0),
}


def _replicates(groups, n_iter, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n_iter):
        pick = rng.choice(ug, len(ug), replace=True)
        yield np.concatenate([idx_by[g] for g in pick])


def cluster_bootstrap(y_true, y_pred, groups, num_classes=9, n_iter=2000, seed=123,
                      alpha=0.05) -> dict:
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    vals = {m: [] for m in METRICS}
    for idx in _replicates(groups, n_iter, seed):
        for m, fn in METRICS.items():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals[m].append(fn(y_true[idx], y_pred[idx], num_classes))
    return {m: {"ci_low": float(np.quantile(v, alpha / 2)), "ci_high": float(np.quantile(v, 1 - alpha / 2)),
                "boot_sd": float(np.std(v))} for m, v in vals.items()} | {
        "method": f"participant cluster bootstrap, {n_iter} iterations, percentile {100*(1-alpha):.0f}% CI",
        "n_clusters": int(len(np.unique(groups)))}


def paired_cluster_bootstrap(y_true, pred_a, pred_b, groups, num_classes=9, n_iter=2000,
                             seed=123, metric="macro_f1") -> dict:
    """CI of metric(a) - metric(b) on the same resampled participants."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    fn = METRICS[metric]
    diffs = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in _replicates(groups, n_iter, seed):
            diffs.append(fn(y_true[idx], pred_a[idx], num_classes) - fn(y_true[idx], pred_b[idx], num_classes))
    diffs = np.asarray(diffs)
    return {"metric": metric, "observed_diff": float(fn(y_true, pred_a, num_classes) - fn(y_true, pred_b, num_classes)),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p_diff_le_0": float((diffs <= 0).mean()), "n_iter": n_iter}
