"""Publication assets: every number, table, figure and manuscript sentence is generated here from the
final machine-readable outputs (prediction files, manifests, efficiency benchmark). Nothing is typed by
hand, so the manuscript cannot drift from the CSVs.

Entry point: ``scripts/make_paper_assets.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from .. import config as C

NC = 9


# ----------------------------------------------------------------------------- data collection
def _metrics(y, p):
    pr, rc, f1, sup = skm.precision_recall_fscore_support(y, p, labels=list(range(NC)), zero_division=0)
    return {"accuracy": skm.accuracy_score(y, p), "macro_precision": pr.mean(), "macro_recall": rc.mean(),
            "macro_f1": f1.mean(), "balanced_accuracy": skm.balanced_accuracy_score(y, p),
            "per_class_precision": pr, "per_class_recall": rc, "per_class_f1": f1, "support": sup,
            "confusion": skm.confusion_matrix(y, p, labels=list(range(NC)))}


def collect(ctx, keys: list[str]) -> dict:
    reg = json.loads((ctx.run_root / "registry.json").read_text())["seeds"]
    runs = {}
    for k in keys:
        for s, rel in sorted(reg.get(k, {}).items(), key=lambda x: int(x[0])):
            d = pd.read_csv(C.REPO_ROOT / rel / "test_predictions.csv").sort_values("video_id").reset_index(drop=True)
            man = json.loads((C.REPO_ROOT / rel / "run_manifest.json").read_text())
            runs[(k, int(s))] = {"pred": d, "metrics": _metrics(d.y_true.values, d.y_pred.values), "manifest": man,
                                 "run_dir": rel}
    return runs


def seed_table(runs, keys):
    rows = []
    for k in keys:
        ss = sorted(s for kk, s in runs if kk == k)
        for s in ss:
            m = runs[(k, s)]["metrics"]
            rows.append({"key": k, "seed": s, **{x: m[x] for x in ["accuracy", "macro_precision", "macro_recall",
                                                                   "macro_f1", "balanced_accuracy"]},
                         "min_class_recall": float(m["per_class_recall"].min())})
    return pd.DataFrame(rows)


def summarise(st: pd.DataFrame) -> pd.DataFrame:
    g = st.groupby("key", sort=False)
    out = g.agg(n_seeds=("seed", "count"), seeds=("seed", lambda x: ",".join(map(str, sorted(x)))))
    for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "min_class_recall"]:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_sd"] = g[m].std(ddof=1)
    return out.reset_index()


def _participant_index(df):
    parts = df.participant.values
    ug = np.unique(parts)
    return ug, {g: np.where(parts == g)[0] for g in ug}


def bootstrap_metric(df, n_iter, seed, fn):
    rng = np.random.default_rng(seed)
    ug, idx = _participant_index(df)
    vals = []
    for _ in range(n_iter):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        vals.append(fn(sel))
    return np.quantile(vals, [0.025, 0.975])


def macro_f1_idx(y, p, sel):
    return skm.f1_score(y[sel], p[sel], labels=list(range(NC)), average="macro", zero_division=0)


def seed_averaged_paired_diff(runs, a, b, seeds, n_iter, boot_seed):
    """Participant-cluster bootstrap of mean_over_seeds[F1(a,s) - F1(b,s)] (same resample for all)."""
    base = runs[(a, seeds[0])]["pred"]
    y = base.y_true.values
    pa = {s: runs[(a, s)]["pred"].y_pred.values for s in seeds}
    pb = {s: runs[(b, s)]["pred"].y_pred.values for s in seeds}
    for s in seeds:  # same videos, same order
        assert (runs[(a, s)]["pred"].video_id.values == base.video_id.values).all()
        assert (runs[(b, s)]["pred"].video_id.values == base.video_id.values).all()
    obs = np.mean([macro_f1_idx(y, pa[s], slice(None)) - macro_f1_idx(y, pb[s], slice(None)) for s in seeds])
    ci = bootstrap_metric(base, n_iter, boot_seed,
                          lambda sel: np.mean([macro_f1_idx(y, pa[s], sel) - macro_f1_idx(y, pb[s], sel) for s in seeds]))
    per_seed = [macro_f1_idx(y, pa[s], slice(None)) - macro_f1_idx(y, pb[s], slice(None)) for s in seeds]
    sd = float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else float("nan")
    if len(seeds) > 1:
        from scipy import stats
        tq = stats.t.ppf(0.975, len(seeds) - 1)
        t_ci = (float(np.mean(per_seed) - tq * sd / math.sqrt(len(seeds))),
                float(np.mean(per_seed) + tq * sd / math.sqrt(len(seeds))))
    else:
        t_ci = (float("nan"), float("nan"))
    return {"a": a, "b": b, "seeds": seeds, "per_seed_diff": per_seed, "mean_diff": float(obs),
            "sd_diff": sd, "frac_seeds_a_better": float(np.mean(np.array(per_seed) > 0)),
            "participant_boot_ci": [float(ci[0]), float(ci[1])], "seed_t_ci": list(t_ci)}


# ----------------------------------------------------------------------------- multi-criteria
CRITERIA = [  # (column, direction, unit)
    ("macro_f1_mean", "max", "macro F1 (mean over seeds)"),
    ("min_class_recall_mean", "max", "lowest per-class recall (mean over seeds)"),
    ("macro_f1_sd", "min", "SD of macro F1 across seeds"),
    ("latency_1t_ms", "min", "median latency, 1 CPU thread (ms/video)"),
    ("peak_rss_mb", "min", "peak resident memory at inference (MB)"),
]
WEIGHT_SCHEMES = {
    "primary (accuracy-led, constrained deployment)": [0.35, 0.20, 0.15, 0.15, 0.15],
    "equal": [0.2, 0.2, 0.2, 0.2, 0.2],
    "performance only": [0.6, 0.4, 0.0, 0.0, 0.0],
    "efficiency-heavy": [0.2, 0.1, 0.1, 0.3, 0.3],
    "stability-heavy": [0.3, 0.2, 0.4, 0.05, 0.05],
}


def multi_criteria(tab: pd.DataFrame, n_random=10000, seed=0):
    t = tab.dropna(subset=[c for c, _, _ in CRITERIA]).copy()
    norm = pd.DataFrame(index=t.index)
    for c, d, _ in CRITERIA:
        lo, hi = t[c].min(), t[c].max()
        v = (t[c] - lo) / (hi - lo) if hi > lo else pd.Series(1.0, index=t.index)
        norm[c] = v if d == "max" else 1 - v
    res = {}
    for name, w in WEIGHT_SCHEMES.items():
        sc = (norm.values * np.array(w)).sum(1)
        res[name] = dict(zip(t.key, sc))
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(CRITERIA)), n_random)
    wins = pd.Series((norm.values @ W.T).argmax(0)).map(dict(enumerate(t.key))).value_counts() / n_random
    return {"normalised": norm.assign(key=t.key.values), "scores": pd.DataFrame(res).rename_axis("key").reset_index(),
            "random_weight_win_rate": wins.to_dict()}


# ----------------------------------------------------------------------------- table writers
def _fmt(v, nd=3):
    if v is None or v is pd.NA or (isinstance(v, float) and (math.isnan(v))):
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def write_table(df: pd.DataFrame, path_stem: Path, caption: str, nd=3, label: str | None = None):
    """CSV (full precision) + Markdown + LaTeX (booktabs)."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    md = [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n")

    def esc(x):
        return (str(x).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
                .replace("_", "\\_").replace("#", "\\#").replace("±", "$\\pm$").replace("–", "--")
                .replace("≥", "$\\geq$").replace("→", "$\\rightarrow$"))
    tex = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{esc(caption)}}}"]
    if label:
        tex.append(f"\\label{{{label}}}")
    tex += ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
            " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        tex.append(" & ".join(esc(_fmt(r[c], nd)) for c in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path_stem.with_suffix(".tex").write_text("\n".join(tex) + "\n")


def pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}" if not (isinstance(s, float) and math.isnan(s)) else f"{m:.{nd}f}"
