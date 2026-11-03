"""Figures (static PNG). Single-hue emphasis for the proposed model, neutral grey for
baselines, sequential single-hue maps for matrices, small multiples - no dual axes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT, NEUTRAL, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def _done(comp: pd.DataFrame) -> pd.DataFrame:
    return comp[comp.status.str.startswith("COMPLETED")].copy()


def model_comparison(comp: pd.DataFrame, out: Path, title_suffix: str = "") -> None:
    d = _done(comp)
    if d.empty:
        return
    d = d.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 1.2))
    colors = [ACCENT if k == "mobiphysio_net" else NEUTRAL for k in d.key]
    y = np.arange(len(d))
    ax.barh(y, d.macro_f1, color=colors, height=0.6)
    err = np.vstack([d.macro_f1 - d.macro_f1_ci_low, d.macro_f1_ci_high - d.macro_f1])
    ax.errorbar(d.macro_f1, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, v, hi in zip(y, d.macro_f1, d.macro_f1_ci_high):
        ax.text(hi + 0.015, yi, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=8)
    ax.set_yticks(y, d.model)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test macro F1 (95% participant-bootstrap CI)")
    ax.set_title(f"Exercise recognition, participant-independent test set{title_suffix}", loc="left", color=INK)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out)
    plt.close(fig)


def confusion_matrices(comp: pd.DataFrame, class_codes, out: Path) -> None:
    d = _done(comp)
    if d.empty:
        return
    n = len(d)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.1 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), d.iterrows()):
        cm = np.array(json.loads(r.confusion_matrix), float)
        norm = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(class_codes)), class_codes, rotation=90, fontsize=6)
        ax.set_yticks(range(len(class_codes)), class_codes, fontsize=6)
        ax.set_title(f"{r.model}  (macro F1 {r.macro_f1:.3f})", fontsize=8, loc="left")
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("true", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (counts; shading = row-normalised recall)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def per_class_f1(per_class: pd.DataFrame, class_codes, out: Path) -> None:
    if per_class.empty:
        return
    piv = per_class.pivot_table(index="model", columns="class_index", values="f1", sort=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(piv) + 1.4))
    ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(piv.shape[1]), [class_codes[c] for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_title("Per-class test F1", loc="left")
    fig.savefig(out)
    plt.close(fig)


def training_curves(histories: dict[str, list], out: Path) -> None:
    hs = {k: h for k, h in histories.items() if h}
    if not hs:
        return
    n = len(hs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False, sharey=True)
    for ax, (name, h) in zip(axes.ravel(), hs.items()):
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["val_macro_f1"] for e in h], color=ACCENT, lw=2, label="val macro F1")
        ax.plot(ep, [e["val_accuracy"] for e in h], color=NEUTRAL, lw=1.5, label="val accuracy")
        best = int(np.argmax([e["val_macro_f1"] for e in h]))
        ax.plot(ep[best], h[best]["val_macro_f1"], "o", color=ACCENT, ms=6, mec="white", mew=2)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(color="#e5e4df", lw=0.6)
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Validation curves (dot = selected checkpoint, max val macro F1)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def dtw_ablation(abl: pd.DataFrame, out: Path) -> None:
    if abl.empty or "macro_f1" not in abl:
        return
    d = abl.dropna(subset=["macro_f1"]).drop_duplicates("key")
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.5 * len(d) + 1.4), sharey=True)
    y = np.arange(len(d))
    colors = [ACCENT if k == "mobiphysio_net" else NEUTRAL for k in d.key]
    axes[0].barh(y, d.macro_f1, color=colors, height=0.6)
    for yi, v in zip(y, d.macro_f1):
        axes[0].text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    axes[0].set_xlim(0, 1.1)
    axes[0].set_xlabel("Test macro F1")
    axes[0].set_yticks(y, d.variant)
    axes[0].invert_yaxis()
    axes[1].barh(y, d.latency_ms_median, color=colors, height=0.6)
    for yi, v in zip(y, d.latency_ms_median):
        axes[1].text(v * 1.01, yi, f"{v:.0f} ms", va="center", fontsize=8, color=INK)
    axes[1].set_xlabel("Median CPU latency per video (ms, incl. ViT)")
    for ax in axes:
        ax.grid(axis="x", color="#e5e4df", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("MobiPhysio-Net ablations: accuracy (left) and cost (right)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
