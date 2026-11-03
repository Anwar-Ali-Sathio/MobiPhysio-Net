"""Classification metrics (Algorithm 8, step 5)."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def classification_metrics(y_true, y_pred, proba=None, num_classes: int = 9,
                           class_names: list[str] | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                     zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(skm.f1_score(y_true, y_pred, labels=labels, average="weighted",
                                          zero_division=0)),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": [{"class_index": k, "class": (class_names[k] if class_names else str(k)),
                       "precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                       "support": int(s[k])} for k in labels],
        "roc_auc_ovr_macro": None, "pr_auc_macro": None, "auc_note": "",
    }
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        present = np.unique(y_true)
        if proba.shape != (len(y_true), num_classes) or not np.isfinite(proba).all():
            out["auc_note"] = "probabilities invalid; AUC not computed"
        elif len(present) < num_classes:
            out["auc_note"] = "not all classes present in y_true; AUC not computed"
        else:
            pr = proba / proba.sum(1, keepdims=True)
            out["roc_auc_ovr_macro"] = float(skm.roc_auc_score(y_true, pr, multi_class="ovr",
                                                              average="macro", labels=labels))
            onehot = np.eye(num_classes)[y_true]
            out["pr_auc_macro"] = float(np.mean([skm.average_precision_score(onehot[:, k], pr[:, k])
                                                 for k in labels]))
            out["auc_note"] = "one-vs-rest macro ROC-AUC; macro average precision (PR-AUC)"
    return out


def macro_f1(y_true, y_pred, num_classes: int = 9) -> float:
    return float(skm.f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro",
                              zero_division=0))


def regression_metrics(y_true, y_pred) -> dict:
    from scipy.stats import pearsonr, spearmanr
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {"n": int(len(y_true)), "mae": float(np.abs(y_true - y_pred).mean()),
           "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
           "r2": float(skm.r2_score(y_true, y_pred))}
    if np.std(y_pred) > 0:
        out["spearman"] = float(spearmanr(y_true, y_pred).statistic)
        out["pearson"] = float(pearsonr(y_true, y_pred).statistic)
    else:
        out["spearman"] = out["pearson"] = None
    return out
