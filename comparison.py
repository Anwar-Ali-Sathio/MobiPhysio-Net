"""Aggregate saved runs into comparison tables (recomputed from prediction files)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import cluster_bootstrap, paired_cluster_bootstrap
from .metrics import classification_metrics
from .predictions import load_predictions

# Final comparison scope lives in configs/experiment.yaml (final_comparison_models); BiLSTM is a
# historical model reported only in results/supplementary/ (decision log D4).
DISPLAY = {"svm": "SVM", "cnn2d": "2D-CNN", "lstm": "LSTM", "bilstm": "BiLSTM", "cnn_lstm": "CNN-LSTM",
           "resnet50": "ResNet50", "mobiphysio_net": "MobiPhysio-Net",
           "mobiphysio_net__no_dtw": "ViT + temporal (no DTW)",
           "mobiphysio_net__dtw_no_temporal": "ViT + DTW (no temporal encoder)",
           "mobiphysio_net__finetuned_vit": "Full, last-block fine-tuned ViT"}
INPUTS = {"svm": "handcrafted HOG + motion-energy descriptor (16 frames)",
          "cnn2d": "RGB 16x112x112, from scratch", "cnn_lstm": "RGB 16x112x112, from scratch",
          "lstm": "frozen ImageNet ResNet50 frame embeddings", "bilstm": "frozen ImageNet ResNet50 frame embeddings",
          "resnet50": "RGB 16x224x224, frozen ImageNet ResNet50 + linear head",
          "mobiphysio_net": "RGB 16x224x224, frozen ViT-S/16 + temporal Transformer + DTW",
          "mobiphysio_net__no_dtw": "RGB 16x224x224, frozen ViT-S/16 + temporal Transformer",
          "mobiphysio_net__dtw_no_temporal": "RGB 16x224x224, frozen ViT-S/16 + DTW on ViT features",
          "mobiphysio_net__finetuned_vit": "RGB 16x224x224, ViT-S/16 last block trained + temporal + DTW"}


def _failed_runs(run_root: Path, key: str) -> list[str]:
    out = []
    for m in (run_root / key).glob("*/run_manifest.json"):
        j = json.loads(m.read_text())
        if j.get("exit_status") == "failed":
            out.append(j.get("error", ""))
    return out


def evaluate_all(repo_root: Path, run_root: Path, res_root: Path, class_codes, class_names,
                 n_boot: int, boot_seed: int, keys: list[str]) -> dict:
    reg_path = run_root / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"primary": {}, "seeds": {}}
    nc = len(class_codes)
    rows, per_class, strat, preds = [], [], [], {}
    for key in keys:
        rel = reg["primary"].get(key)
        if not rel:
            failed = _failed_runs(run_root, key)
            rows.append({"key": key, "model": DISPLAY.get(key, key), "input": INPUTS.get(key, ""),
                         "status": "FAILED" if failed else "NOT_EXECUTED",
                         "error": failed[-1] if failed else ""})
            continue
        run_dir = repo_root / rel
        df, proba = load_predictions(run_dir / "test_predictions.csv", class_codes)
        stored = json.loads((run_dir / "metrics.json").read_text())
        man = json.loads((run_dir / "run_manifest.json").read_text())
        m = classification_metrics(df.y_true, df.y_pred, proba, nc, class_names)
        ci = cluster_bootstrap(df.y_true, df.y_pred, df.participant, nc, n_boot, boot_seed)
        preds[key] = df
        rows.append({"key": key, "model": DISPLAY.get(key, key), "input": INPUTS.get(key, ""),
                     "status": "COMPLETED" + (" (PILOT)" if man.get("pilot") else ""),
                     "run_id": man["run_id"], "seed": man["seed"],
                     "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                     "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                     "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                     "macro_f1_ci_low": ci["macro_f1"]["ci_low"], "macro_f1_ci_high": ci["macro_f1"]["ci_high"],
                     "accuracy_ci_low": ci["accuracy"]["ci_low"], "accuracy_ci_high": ci["accuracy"]["ci_high"],
                     "balanced_accuracy_ci_low": ci["balanced_accuracy"]["ci_low"],
                     "balanced_accuracy_ci_high": ci["balanced_accuracy"]["ci_high"],
                     "roc_auc_ovr_macro": m["roc_auc_ovr_macro"], "pr_auc_macro": m["pr_auc_macro"],
                     "test_videos": m["n"], "test_participants": int(df.participant.nunique()),
                     "latency_ms_median": stored["latency"]["median_ms"],
                     "latency_ms_p90": stored["latency"]["p90_ms"],
                     "params_trainable": stored["params_trainable"], "params_total": stored["params_total"],
                     "best_epoch": man.get("best_epoch"), "best_val_macro_f1": man.get("best_val_macro_f1"),
                     "confusion_matrix": json.dumps(m["confusion_matrix"])})
        for pc in m["per_class"]:
            per_class.append({"key": key, "model": DISPLAY.get(key, key), **pc})
        for col in ["angle", "variation"]:
            if col in df:
                for v, g in df.groupby(col):
                    mm = classification_metrics(g.y_true, g.y_pred, None, nc)
                    strat.append({"key": key, "model": DISPLAY.get(key, key), "factor": col, "level": v,
                                  "n": len(g), "accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]})
        shutil.copyfile(run_dir / "test_predictions.csv", res_root / "predictions" / f"{key}_test_predictions.csv")
    comp = pd.DataFrame(rows)
    # Seed variability
    seeds = []
    for key, runs in reg.get("seeds", {}).items():
        if key not in keys:
            continue
        vals = []
        for s, rel in sorted(runs.items()):
            d, p = load_predictions(repo_root / rel / "test_predictions.csv", class_codes)
            mm = classification_metrics(d.y_true, d.y_pred, p, nc)
            vals.append({"seed": int(s), "macro_f1": mm["macro_f1"], "accuracy": mm["accuracy"]})
        f1 = [v["macro_f1"] for v in vals]
        seeds.append({"key": key, "model": DISPLAY.get(key, key), "n_seeds": len(vals),
                      "seeds": json.dumps([v["seed"] for v in vals]),
                      "macro_f1_mean": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else None,
                      "accuracy_mean": float(np.mean([v["accuracy"] for v in vals])),
                      "per_seed_macro_f1": json.dumps([round(x, 4) for x in f1])})
    return {"comparison": comp, "per_class": pd.DataFrame(per_class), "stratified": pd.DataFrame(strat),
            "seeds": pd.DataFrame(seeds), "predictions": preds}


def ablation_table(preds: dict, comp: pd.DataFrame, n_boot: int, seed: int, nc: int) -> pd.DataFrame:
    """Each variant vs full MobiPhysio-Net on identical test videos (paired bootstrap)."""
    full = preds.get("mobiphysio_net")
    rows = []
    if full is None:
        return pd.DataFrame()
    cm = comp.set_index("key")
    variants = [("mobiphysio_net", "Full MobiPhysio-Net (frozen ViT)"),
                ("mobiphysio_net__no_dtw", "ViT + temporal encoder, no DTW"),
                ("mobiphysio_net__dtw_no_temporal", "ViT + DTW features, no temporal encoder"),
                ("mobiphysio_net", "Full model, frozen ViT (same run as row 1)"),
                ("mobiphysio_net__finetuned_vit", "Full model, last-block fine-tuned ViT")]
    for key, label in variants:
        if key not in preds:
            rows.append({"variant": label, "key": key, "status": cm.loc[key, "status"] if key in cm.index else "NOT_EXECUTED"})
            continue
        d = preds[key].set_index("video_id").loc[full.video_id]
        if not (d.y_true.values == full.y_true.values).all():
            raise ValueError(f"{key}: test videos/labels differ from full model")
        r = cm.loc[key]
        row = {"variant": label, "key": key, "status": r["status"], "macro_f1": r["macro_f1"],
               "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
               "latency_ms_median": r["latency_ms_median"], "params_trainable": r["params_trainable"],
               "params_total": r["params_total"]}
        if key != "mobiphysio_net":
            pb = paired_cluster_bootstrap(full.y_true.values, full.y_pred.values, d.y_pred.values,
                                          full.participant.values, nc, n_boot, seed)
            row.update({"full_minus_variant_macro_f1": pb["observed_diff"],
                        "diff_ci_low": pb["ci_low"], "diff_ci_high": pb["ci_high"],
                        "latency_full_minus_variant_ms": cm.loc["mobiphysio_net", "latency_ms_median"] - r["latency_ms_median"],
                        "mcnemar_b_full_right_variant_wrong": int(((full.y_pred.values == full.y_true.values) & (d.y_pred.values != d.y_true.values)).sum()),
                        "mcnemar_c_full_wrong_variant_right": int(((full.y_pred.values != full.y_true.values) & (d.y_pred.values == d.y_true.values)).sum())})
        rows.append(row)
    return pd.DataFrame(rows)
