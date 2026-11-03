"""Determinism check: retrain selected (model, seed) pairs with the stored configuration and compare the new
test predictions with the registered runs.

Usage:  PYTHONPATH=src python scripts/determinism_check.py [key:seed ...]   (default: svm:42 mobiphysio_net:42)
Writes runs to experiments/determinism_check_<exp>/ (never touches the reported runs or their registry) and
reports/determinism_check.{json,md}.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mobiphysio import pipeline as P  # noqa: E402


class CheckContext(P.Context):
    @property
    def run_root(self) -> Path:
        p = ROOT / "experiments" / f"determinism_check_{self.exp_cfg['name']}"
        p.mkdir(parents=True, exist_ok=True)
        return p


def main(pairs):
    base = P.Context.create(None, "experiment", {})
    ctx = CheckContext(**{f: getattr(base, f) for f in base.__dataclass_fields__})
    orig_root = ROOT / "experiments" / base.exp_cfg["name"]
    # reuse the reported reference bank (loaded, not rebuilt) so the DTW inputs are identical
    shutil.copy2(orig_root / "reference_bank.json", ctx.run_root / "reference_bank.json")
    reg = json.loads((orig_root / "registry.json").read_text())["seeds"]
    out = []
    for key, seed in pairs:
        ref = ROOT / reg[key][str(seed)]
        r = P.train_one(ctx, key, seed)
        if r.get("exit_status") == "failed":
            out.append({"key": key, "seed": seed, "status": "failed", "error": r["error"]})
            continue
        a = pd.read_csv(ref / "test_predictions.csv").sort_values("video_id").reset_index(drop=True)
        b = pd.read_csv(Path(r["run_dir"]) / "test_predictions.csv").sort_values("video_id").reset_index(drop=True)
        pc = [c for c in a.columns if c.startswith("p_") or c.startswith("prob")]
        same_ids = bool((a.video_id.values == b.video_id.values).all())
        agree = float((a.y_pred.values == b.y_pred.values).mean())
        dmax = float(np.abs(a[pc].to_numpy(float) - b[pc].to_numpy(float)).max()) if pc else float("nan")
        f_ref = json.loads((ref / "metrics.json").read_text())["macro_f1"]
        out.append({"key": key, "seed": seed, "status": "completed", "reference_run": str(ref.relative_to(ROOT)),
                    "rerun": str(Path(r["run_dir"]).relative_to(ROOT)), "same_test_videos": same_ids,
                    "prediction_agreement": agree, "max_abs_probability_diff": dmax,
                    "macro_f1_reference": f_ref, "macro_f1_rerun": r["test_macro_f1"],
                    "bitwise_identical_predictions": agree == 1.0 and dmax == 0.0})
    rep = ROOT / "reports"
    (rep / "determinism_check.json").write_text(json.dumps(out, indent=1))
    lines = ["# Determinism check", "", "Selected runs were retrained from scratch with the stored seed and configuration "
             "(same split, same cached inputs, same reference bank) by `scripts/determinism_check.py`, and the new "
             "per-video test predictions were compared with the registered runs. Reruns are kept in "
             f"`experiments/determinism_check_{base.exp_cfg['name']}/` and are not part of any reported result.", "",
             "| Model | Seed | Same test videos | Prediction agreement | Max |Δp| | Macro F1 reported | Macro F1 rerun | Identical |",
             "|---|---|---|---|---|---|---|---|"]
    for o in out:
        if o["status"] != "completed":
            lines.append(f"| {o['key']} | {o['seed']} | — | — | — | — | — | FAILED: {o['error']} |")
            continue
        lines.append(f"| {o['key']} | {o['seed']} | {o['same_test_videos']} | {o['prediction_agreement']:.4f} | "
                     f"{o['max_abs_probability_diff']:.2e} | {o['macro_f1_reference']:.4f} | {o['macro_f1_rerun']:.4f} | "
                     f"{'yes' if o['bitwise_identical_predictions'] else 'no'} |")
    lines += ["", "Environment: same machine and software versions as the reported runs (CPU only). Identity across other "
              "hardware or library versions is not claimed."]
    (rep / "determinism_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    args = sys.argv[1:] or ["svm:42", "mobiphysio_net:42"]
    main([(a.split(":")[0], int(a.split(":")[1])) for a in args])
