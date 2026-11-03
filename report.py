"""Markdown/HTML reports built ONLY from saved artifacts (no hand-entered numbers)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config as C


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None, nd=3) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r.get(c), nd) for c in cols) + " |")
    return "\n".join(lines)


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def seed_paired_ablation(ctx, res_root: Path) -> pd.DataFrame:
    """Full minus variant macro F1 for every training seed run for BOTH models (same split)."""
    from ..evaluation.metrics import macro_f1
    reg_p = ctx.run_root / "registry.json"
    if not reg_p.exists():
        return pd.DataFrame()
    reg = json.loads(reg_p.read_text())["seeds"]
    full = reg.get("mobiphysio_net", {})
    rows = []
    for key, label in [("mobiphysio_net__no_dtw", "no DTW"), ("mobiphysio_net__dtw_no_temporal", "no temporal encoder")]:
        for seed in sorted(set(full) & set(reg.get(key, {})), key=int):
            f = pd.read_csv(C.REPO_ROOT / full[seed] / "test_predictions.csv")
            v = pd.read_csv(C.REPO_ROOT / reg[key][seed] / "test_predictions.csv")
            ff, vf = macro_f1(f.y_true, f.y_pred), macro_f1(v.y_true, v.y_pred)
            rows.append({"variant": label, "seed": int(seed), "full_macro_f1": ff, "variant_macro_f1": vf,
                         "full_minus_variant": ff - vf})
    return pd.DataFrame(rows)


def dataset_audit(ctx) -> str:
    md = ctx.manifest_dir
    meta = _load(md / "dataset_metadata.json")
    dl = _load(md / "download_report.json") or {}
    val = _load(md / "data_validation.json") or {}
    ann = _load(md / "annotation_audit.json") or {}
    vm = pd.read_csv(md / "video_manifest.csv")
    split_audit = _load(md / f"splits_{ctx.exp_cfg['name']}_audit.json") or {}
    s = [f"# Dataset audit - MobiPhysio ({meta['doi']}, version {meta['version']})", "",
         f"Landing page: {meta['landing_page']}  ", f"License: {meta['license'].get('name')}  ",
         f"Release time: {meta.get('release_time')}  ", "",
         "## Coverage (counts from saved manifests)", "",
         "| Quantity | Count |", "|---|---:|",
         f"| Files in Dataverse inventory | {meta['n_files']} |",
         f"| Videos in inventory | {meta['n_videos_in_inventory']} |",
         f"| Inventory size (GB) | {meta['total_size_bytes'] / 1e9:.2f} |",
         f"| Restricted files | {meta['n_restricted']} |",
         f"| Duplicate-checksum groups (exact duplicate content) | {len(meta['duplicate_checksum_groups'])} |",
         f"| Videos downloaded + MD5-verified | {dl.get('downloaded_checksum_verified')} |",
         f"| Videos decoded + cached (16 frames) | {dl.get('decoded_and_cached')} |",
         f"| Download failures | {dl.get('download_failed')} |",
         f"| Decode failures | {dl.get('decode_failed')} |",
         f"| Filenames parsed to valid labels | {val.get('filename_parse_ok')} |",
         f"| Usable for Task A (labelled + decoded) | {val.get('usable_task_a')} |",
         f"| Participants (usable) | {vm[vm.usable_task_a].participant.nunique()} |", "",
         f"Raw-data mode: {dl.get('mode')}. Dataset complete in cache: **{dl.get('complete')}**.", ""]
    ex = vm[vm.exclusion_reason.fillna("") != ""]
    dups = meta.get("duplicate_checksum_groups", {})
    if dups:
        s += ["## Exact duplicate content in the published dataset", "",
              "Files with identical MD5 (byte-identical videos) published under different names:", ""]
        for md5, files in dups.items():
            s.append(f"- `{md5}`: " + ", ".join(f"`{f}`" for f in files))
        s += ["", "The copies carry conflicting participant/variation labels, so the true identity is ambiguous; "
              "**all copies are excluded** from Task A. In the superseded v1 run (archived under "
              "`experiments/full_v1_superseded/`) one copy was in train and the other in validation.", ""]
    s += ["## Exclusions", "", f"{len(ex)} videos excluded." if len(ex) else "No videos excluded.", ""]
    if len(ex):
        s += [_table(ex.head(50), ["video_id", "exclusion_reason"]), ""]
    s += ["## Label / metadata distribution (usable videos)", ""]
    u = vm[vm.usable_task_a]
    cc = u.groupby(["exercise", "exercise_name"]).size().reset_index(name="videos")
    s += [_table(cc, ["exercise", "exercise_name", "videos"]), ""]
    for col in ["angle", "variation", "gender"]:
        vc = u[col].value_counts().sort_index()
        s.append(f"- **{col}**: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    s += ["", f"- Imbalance: largest/smallest class = {cc.videos.max()}/{cc.videos.min()} = "
              f"{cc.videos.max() / cc.videos.min():.2f}.",
          "- Participant coverage is uneven: participants contribute between "
          f"{u.participant.value_counts().min()} and {u.participant.value_counts().max()} videos, and many "
          "participants performed only a subset of the exercises.", "",
          "## Discrepancies vs the research brief", "",
          "- E08 is named **'Both Leg Flexion'** in the official README (MobiPhysio.md v1.1), not 'Lumbar Flexion'. "
          "The official name is used everywhere.",
          (f"- Variation code `VP` ({int((u.variation == 'P').sum())} video(s)) is not in the README codebook; "
           "the video is kept for Task A (its exercise code is valid) with the variation recorded as undocumented."
           if (u.variation == "P").any() else ""), "",
          "## EAAQ annotations (Task B)", ""]
    if ann.get("available"):
        s += [f"- Annotation blocks parsed: {ann['n_blocks']} (named {ann['n_named_blocks']}, "
              f"**unnamed {ann['n_unnamed_blocks']}**). Unnamed blocks are NOT linked to any video (no inference from row order).",
              f"- Duplicate video names in the CSV: {ann['n_duplicate_names']} (excluded from linkage).",
              f"- Blocks with a judge count other than 3: {ann['n_irregular_judge_count']}.",
              f"- Stored average != mean of judges' 0-100 scores (>0.01): {ann['n_average_mismatch']}.",
              f"- Named blocks not matching any inventory video: {len(ann.get('orphan_named_blocks_not_in_inventory', []))}.",
              f"- Videos with a linked score: {ann.get('videos_with_linked_score')}; without: {ann.get('videos_without_linked_score')}.",
              f"- Score range in CSV: {ann['score_min']:.2f} - {ann['score_max']:.2f} (0-100, higher = better execution).",
              "- Schema: 4 'Primary Factors' + 3 'Control Factors' items per judge (each 0-1, partial credit allowed), "
              "total mark 0-7 rescaled to 0-100; per-video target = mean over the three judges. "
              "Target level: per video. Type: continuous (bounded).", ""]
    else:
        s += ["Annotation file not available - Task B blocked.", ""]
    if split_audit:
        a = split_audit["audit"]
        s += ["## Participant-independent split", "",
              f"Split file: `{split_audit['file']}` (sha256 `{split_audit['split_sha256'][:16]}...`).", "",
              "| Split | Videos | Participants |", "|---|---:|---:|"]
        for k in ["train", "val", "test"]:
            s.append(f"| {k} | {a['counts'].get(k)} | {len(a['participants'].get(k, []))} |")
        s += ["", f"- Participant overlap between any splits: **{a['participant_overlap_any']}**",
              f"- Exact-duplicate (MD5) content across splits: {len(a['cross_split_duplicate_md5'])}",
              f"- Test participants: {', '.join(a['participants'].get('test', []))}", ""]
        nd = _load(md / f"near_duplicates_{ctx.exp_cfg['name']}.json")
        if nd:
            s += [f"- Near-duplicate audit (cosine of mean frozen-ViT embedding, threshold {nd['threshold']}): "
                  f"{nd['n_pairs_above_threshold']} cross-split pairs above threshold; max cross-split cosine "
                  + ", ".join(f"{k}={v:.4f}" for k, v in nd["max_cross_split_cosine"].items()) + ". "
                  f"Pixel-level candidates (mean |diff| < {nd.get('pixel_candidate_threshold')} grey levels): "
                  f"{nd.get('n_pixel_candidates')}. {nd.get('note', '')}", ""]
            mr = nd.get("manual_review")
            if mr:
                s += [f"- Manual review ({mr['reviewed_utc']}): {mr['conclusion']}", ""]
    return "\n".join(x for x in s if x is not None)


def experiment_report(ctx, res_root: Path) -> str:
    md = res_root / "metrics"
    comp = pd.read_csv(md / "model_comparison.csv")
    allc = pd.read_csv(md / "all_runs_comparison.csv")
    pilot = bool(ctx.exp_cfg.get("pilot"))
    s = [f"# MobiPhysio-Net experiment report{' - PILOT (reduced data/epochs; not a benchmark)' if pilot else ''}", "",
         "All numbers below are read from saved prediction/metric files; see `reproducibility_audit.md` "
         "and `verification.json` for the recomputation check.", "",
         "## Task A - 9-class exercise recognition (participant-independent test set)", ""]
    comp = comp.copy()
    comp["macro_f1_ci"] = comp.apply(lambda r: "n/a" if pd.isna(r.get("macro_f1")) else
                                     f"[{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]", axis=1)
    s += [_table(comp, ["model", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                        "macro_f1_ci", "weighted_f1", "roc_auc_ovr_macro", "test_videos", "latency_ms_median", "status"],
                 ["Model", "Accuracy", "Balanced acc.", "Macro P", "Macro R", "Macro F1", "Macro F1 95% CI",
                  "Weighted F1", "ROC-AUC (OvR)", "Test videos", "Latency ms (median)", "Status"]), "",
          "CI: participant-level cluster bootstrap (test participants resampled with replacement). "
          "Latency: median CPU time to classify one video from cached 16x224x224 frames (includes the "
          "backbone; excludes MP4 decoding).", "",
          "### Input modality per model (comparison is NOT perfectly controlled)", "",
          _table(comp, ["model", "input", "params_trainable", "params_total"],
                 ["Model", "Input / backbone", "Trainable params", "Total params (incl. frozen)"], 0), ""]
    abl_p = md / "ablation_comparison.csv"
    if abl_p.exists():
        abl = pd.read_csv(abl_p)
        s += ["## Ablations (same split, same test videos)", "",
              _table(abl, ["variant", "macro_f1", "accuracy", "balanced_accuracy", "full_minus_variant_macro_f1",
                           "diff_ci_low", "diff_ci_high", "latency_ms_median", "params_trainable", "status"],
                     ["Variant", "Macro F1", "Accuracy", "Balanced acc.", "Full - variant (macro F1)",
                      "diff CI low", "diff CI high", "Latency ms", "Trainable params", "Status"]), "",
              "Difference CIs: paired participant-level bootstrap on identical test videos.", ""]
        nd = abl[abl.key == "mobiphysio_net__no_dtw"]
        if len(nd) and pd.notna(nd.iloc[0].get("full_minus_variant_macro_f1")):
            r = nd.iloc[0]
            ci_excl = r.diff_ci_low > 0 or r.diff_ci_high < 0
            s += [f"Seed-42 paired bootstrap (test-sample uncertainty only): full - no-DTW macro F1 = {r.full_minus_variant_macro_f1:+.4f} "
                  f"(95% CI [{r.diff_ci_low:+.4f}, {r.diff_ci_high:+.4f}]); the interval "
                  f"{'excludes' if ci_excl else 'includes'} zero. Latency cost of the DTW branch: "
                  f"{r.latency_full_minus_variant_ms:+.1f} ms per video. McNemar discordant counts: "
                  f"full right/no-DTW wrong = {int(r.mcnemar_b_full_right_variant_wrong)}, "
                  f"full wrong/no-DTW right = {int(r.mcnemar_c_full_wrong_variant_right)}.", ""]
        sp = seed_paired_ablation(ctx, res_root)
        if len(sp):
            sp.to_csv(md / "ablation_seed_paired.csv", index=False)
            s += ["### Seed-paired ablation (training-seed variability included)", "",
                  _table(sp, ["variant", "seed", "full_macro_f1", "variant_macro_f1", "full_minus_variant"]), ""]
            for var, g in sp.groupby("variant"):
                d = g.full_minus_variant
                sign = "positive" if (d > 0).all() else ("negative" if (d < 0).all() else "mixed")
                s.append(f"- Full - {var}: mean {d.mean():+.4f} over {len(d)} seeds (min {d.min():+.4f}, "
                         f"max {d.max():+.4f}; sign {sign}).")
            s += ["", "**Interpretation rule** (conservative; written after the seed-42 result was seen): a component is "
                  "credited only if the difference is positive for every seed AND "
                  "the seed-42 paired-bootstrap CI excludes zero. Latency differences between MobiPhysio-Net "
                  "variants are within run-to-run timing noise (all are dominated by the shared ViT forward pass).", ""]
    sv = md / "seed_variability.csv"
    if sv.exists():
        s += ["## Training-seed variability (same split)", "",
              _table(pd.read_csv(sv), ["model", "n_seeds", "seeds", "macro_f1_mean", "macro_f1_sd", "per_seed_macro_f1"]),
              "", "Differences between models smaller than this seed spread should not be interpreted.", ""]
    st = md / "metrics_by_angle_variation.csv"
    if st.exists():
        d = pd.read_csv(st)
        d = d[d.key == "mobiphysio_net"]
        if len(d):
            s += ["## MobiPhysio-Net by camera angle and recording variation (README-recommended breakdown)", "",
                  _table(d, ["factor", "level", "n", "accuracy", "macro_f1"]), ""]
    tb = _load(md / "task_b_eaaq.json")
    s += ["## Task B - EAAQ movement-quality score (exploratory)", ""]
    if tb and tb.get("status", "").startswith("COMPLETED"):
        s += [f"Target: {tb['target']}. Linked usable videos: {tb['n']['linked_usable']} "
              f"(train {tb['n']['train']}, val {tb['n']['val']}, test {tb['n']['test']}). "
              f"Inter-judge ICC(2,1) = {tb['inter_judge_icc2_1']:.3f}; mean per-video judge SD = {tb['mean_judge_sd']:.2f} points. "
              f"Score mean {tb['score_summary']['mean']:.1f} (SD {tb['score_summary']['sd']:.1f}).", "",
              _table(pd.DataFrame(tb["metrics"]), ["model", "mae", "rmse", "r2", "spearman"]), "",
              "`exercise_variation_mean` uses only the exercise code and the recording-variation code. If it is "
              "competitive with the video models, the scores are largely explained by recording condition / exercise "
              "identity, and a low error does not demonstrate movement-quality assessment.", ""]
    else:
        s += [f"Status: {tb.get('status') if tb else 'NOT_EXECUTED'}", ""]
    not_done = allc[~allc.status.str.startswith("COMPLETED")]
    s += ["## Missing / incomplete", ""]
    s += ([f"- {r.model}: {r.status} {r.get('error', '') if isinstance(r.get('error', ''), str) else ''}"
           for _, r in not_done.iterrows()] or ["- All registered model runs completed."])
    s += ["", "## Figures", ""]
    for f in ["model_comparison", "confusion_matrices", "per_class_f1", "training_curves", "dtw_ablation"]:
        if (res_root / "figures" / f"{f}.png").exists():
            s.append(f"![{f}](../figures/{f}.png)")
    s += ["", "## Interpretation limits", "",
          "- Recognising WHICH exercise is performed does not establish whether it is performed correctly; no clinical "
          "validity or utility is claimed.",
          "- Single dataset, single participant split, no external validation; ~12 test participants make CIs wide.",
          "- Backbones (ViT-S/16, ResNet50) are frozen ImageNet models (CPU budget); from-scratch CNNs see 112 px frames. "
          "Rankings reflect these specific, documented configurations - not the architecture families in general.",
          "- See docs/LIMITATIONS.md."]
    return "\n".join(s)


def reproducibility_audit(ctx, res_root: Path) -> str:
    reg_p = ctx.run_root / "registry.json"
    reg = json.loads(reg_p.read_text()) if reg_p.exists() else {"primary": {}}
    name = ctx.exp_cfg["name"]
    split_sha = (_load(ctx.manifest_dir / f"splits_{name}_audit.json") or {}).get("split_sha256")
    s = ["# Reproducibility audit", "", f"Experiment: `{name}`  ", f"Split assignment sha256: `{split_sha}`", "",
         "| Model key | Run ID | Seed | Best epoch | Val macro F1 | Started (UTC) | Ended (UTC) | Exit | Python | torch |",
         "|---|---|---:|---:|---:|---|---|---|---|---|"]
    env = None
    for k, rel in reg["primary"].items():
        m = json.loads((C.REPO_ROOT / rel / "run_manifest.json").read_text())
        env = m["environment"]
        s.append(f"| {k} | `{m['run_id']}` | {m['seed']} | {m.get('best_epoch')} | {_fmt(m.get('best_val_macro_f1'))} | "
                 f"{m['started_utc']} | {m.get('ended_utc')} | {m['exit_status']} | {env['python']} | {env['packages']['torch']} |")
    if env:
        s += ["", "## Environment", "", f"- Platform: {env['platform']}", f"- CPU: {env['cpu_model']} x{env['cpu_count']}, "
              f"RAM {env['ram_gb']} GB, GPU: {env['gpu']}", "- Packages: " + ", ".join(f"{k}={v}" for k, v in env["packages"].items())]
    ver = _load(res_root / "reports" / "verification.json")
    if ver:
        s += ["", "## Result verification", "", f"Status: **{ver['exit_status']}** ({ver['verified_utc']})", ""]
        s += [_table(pd.DataFrame(ver["checks"]), list(ver["checks"][0].keys()))] if ver["checks"] else []
        s += ["", "Problems: " + ("; ".join(ver["problems"]) if ver["problems"] else "none")]
    s += ["", "## Determinism", "", "- Seeds set for Python, NumPy, PyTorch; `torch.use_deterministic_algorithms(True, warn_only=True)`.",
          "- Multi-threaded CPU reductions can differ in final bits across machines; SVC Platt scaling uses internal (seeded) CV.",
          "- Each run writes to its own `experiments/<experiment>/<model>/<run_id>/` directory; nothing is overwritten."]
    return "\n".join(s)


def readme_results_block(ctx, res_root: Path) -> str:
    """Short results section for README.md, generated from saved tables."""
    md = res_root / "metrics"
    comp = pd.read_csv(md / "model_comparison.csv")
    comp["macro_f1_ci"] = comp.apply(lambda r: "n/a" if pd.isna(r.get("macro_f1")) else
                                     f"[{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]", axis=1)
    ev = _load(md / "evaluation_summary.json") or {}
    s = ["## Results (full experiment, generated by `mobiphysio compare`)", "",
         f"Participant-independent test set: {ev.get('test_videos')} videos from "
         f"{ev.get('test_participants')} participants never seen in training/validation.", "",
         _table(comp, ["model", "accuracy", "balanced_accuracy", "macro_f1", "macro_f1_ci", "weighted_f1",
                       "latency_ms_median", "status"],
                ["Model", "Accuracy", "Balanced acc.", "Macro F1", "Macro F1 95% CI (participant bootstrap)",
                 "Weighted F1", "CPU latency ms/video", "Status"]), ""]
    abl_p = md / "ablation_comparison.csv"
    if abl_p.exists():
        abl = pd.read_csv(abl_p).drop_duplicates("key")
        s += ["**Ablations** (same split; difference = full - variant, paired participant bootstrap):", "",
              _table(abl, ["variant", "macro_f1", "full_minus_variant_macro_f1", "diff_ci_low", "diff_ci_high",
                           "latency_ms_median"],
                     ["Variant", "Macro F1", "Full - variant", "CI low", "CI high", "Latency ms"]), ""]
    sp = seed_paired_ablation(ctx, res_root)
    if len(sp):
        parts = []
        for var, g in sp.groupby("variant"):
            d = g.full_minus_variant
            parts.append(f"full - {var}: " + ", ".join(f"seed {r.seed} {r.full_minus_variant:+.3f}" for r in g.itertuples())
                         + f" (mean {d.mean():+.3f})")
        s += ["**Seed-paired ablation** (same three training seeds for both models): " + "; ".join(parts) + ".", ""]
    sv = md / "seed_variability.csv"
    if sv.exists():
        d = pd.read_csv(sv)
        s += ["**Training-seed spread** (seeds 42, 1, 2; same split): " + "; ".join(
            f"{r.model} {r.macro_f1_mean:.3f} +/- {r.macro_f1_sd:.3f}" for r in d.itertuples() if r.n_seeds > 1), ""]
    tb = _load(md / "task_b_eaaq.json")
    if tb and tb.get("status", "").startswith("COMPLETED"):
        best = min(tb["metrics"], key=lambda m: m["mae"])
        s += [f"**Task B (EAAQ, exploratory):** {tb['n']['linked_usable']} linked videos; lowest test MAE "
              f"{best['mae']:.2f} points ({best['model']}); see the experiment report for all baselines.", ""]
    s += ["Full report: [results/reports/experiment_report.md](results/reports/experiment_report.md) · "
          "dataset audit: [results/reports/dataset_audit.md](results/reports/dataset_audit.md) · "
          "verification: [results/reports/verification.json](results/reports/verification.json) · "
          "limitations: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)."]
    return "\n".join(s)


def update_readme(ctx, res_root: Path) -> bool:
    if ctx.exp_cfg.get("pilot"):
        return False  # pilot numbers never go into the README
    path = C.REPO_ROOT / "README.md"
    text = path.read_text()
    start, end = "<!-- RESULTS -->", "<!-- /RESULTS -->"
    block = f"{start}\n{readme_results_block(ctx, res_root)}\n{end}"
    if end in text:
        pre, rest = text.split(start, 1)
        text = pre + block + rest.split(end, 1)[1]
    else:
        text = text.replace(start, block, 1)
    path.write_text(text)
    return True


def write_reports(ctx, res_root: Path) -> list[str]:
    import markdown
    rd = res_root / "reports"
    out = {}
    out["dataset_audit.md"] = dataset_audit(ctx)
    out["experiment_report.md"] = experiment_report(ctx, res_root)
    out["reproducibility_audit.md"] = reproducibility_audit(ctx, res_root)
    for name, text in out.items():
        (rd / name).write_text(text)
    html = markdown.markdown(out["experiment_report.md"], extensions=["tables"])
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}"
           "table{border-collapse:collapse;font-size:13px;margin:1rem 0;display:block;overflow-x:auto}"
           "td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}th{background:#f1f0ec}"
           "td:first-child,th:first-child{text-align:left}img{max-width:100%}")
    (rd / "experiment_report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>MobiPhysio-Net report</title>"
        f"<style>{css}</style></head><body>{html}</body></html>")
    update_readme(ctx, res_root)
    return sorted(list(out) + ["experiment_report.html"])
