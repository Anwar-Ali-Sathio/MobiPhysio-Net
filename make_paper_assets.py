"""Generate paper_assets/ (tables, figures, manuscript text) from the final machine-readable outputs.

Usage:  PYTHONPATH=src python scripts/make_paper_assets.py
Inputs: experiments/<exp>/registry.json + run dirs (predictions, manifests), data/manifests/*,
        results/efficiency/* (from scripts/benchmark_efficiency.py), results/metrics/task_b_eaaq.json.
Output: paper_assets/{data/final_numbers.json, tables/*.{csv,md,tex}, figures/*.{png,pdf},
        manuscript/*.{md,tex}, FIGURE_INDEX.md, TABLE_INDEX.md}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mobiphysio import pipeline as P  # noqa: E402
from mobiphysio.evaluation.comparison import DISPLAY  # noqa: E402
from mobiphysio.reporting import paper as R  # noqa: E402

OUT = Path(os.environ.get("MOBIPHYSIO_PAPER_OUT", ROOT / "paper_assets"))
FIG, TAB, MAN, DAT = OUT / "figures", OUT / "tables", OUT / "manuscript", OUT / "data"
CMD = "PYTHONPATH=src python scripts/make_paper_assets.py"
ACCENT, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e5e4df"
ABL_LABEL = {"mobiphysio_net": "A. Full (frozen ViT + temporal + DTW)",
             "mobiphysio_net__no_dtw": "B. No DTW",
             "mobiphysio_net__dtw_no_temporal": "C. No temporal encoder",
             "mobiphysio_net__finetuned_vit": "D. Last-block fine-tuned ViT"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def main():
    ctx = P.Context.create(None, "experiment", {})
    finals, abls, hist = P.final_models(ctx), P.ablation_keys(ctx), P.historical_models(ctx)
    codes = sorted(ctx.dataset_cfg["exercises"])
    names = [ctx.dataset_cfg["exercises"][c] for c in codes]
    ev = ctx.exp_cfg["evaluation"]
    nb, bseed = ev["bootstrap_iterations"], ev["bootstrap_seed"]
    seed0 = ctx.exp_cfg["seed"]
    runs = R.collect(ctx, finals + abls + hist)
    st = R.seed_table(runs, finals + abls + hist)
    sm = R.summarise(st).set_index("key")
    N = {"experiment": ctx.exp_cfg["name"], "primary_seed": seed0, "bootstrap_iterations": nb}

    # ---------------------------------------------------------------- dataset (Table 1)
    md = ctx.manifest_dir
    meta = json.loads((md / "dataset_metadata.json").read_text())
    vm = pd.read_csv(md / "video_manifest.csv")
    sp = pd.read_csv(md / f"splits_{ctx.exp_cfg['name']}.csv")
    N["inventory_files"], N["inventory_videos"] = meta["n_files"], meta["n_videos_in_inventory"]
    N["dataset_version"], N["usable_videos"] = meta["version"], int(vm.usable_task_a.sum())
    N["excluded_duplicates"] = int(vm.exclusion_reason.fillna("").str.startswith("duplicate").sum())
    N["participants_total"] = int(sp.participant.nunique())
    use = vm[vm.usable_task_a]
    N["median_duration_s"] = float((use.n_frames / use.fps).median())
    for s in ["train", "val", "test"]:
        N[f"n_{s}"] = int((sp.split == s).sum())
        N[f"p_{s}"] = int(sp[sp.split == s].participant.nunique())
    pp = {s: set(sp[sp.split == s].participant) for s in ["train", "val", "test"]}
    N["participant_overlap_pairs"] = sum(len(pp[a] & pp[b]) for a in pp for b in pp if a < b)
    t1 = sp.groupby(["exercise", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    t1["total"] = t1.sum(axis=1)
    t1.insert(0, "name", [ctx.dataset_cfg["exercises"][e] for e in t1.index])
    t1 = t1.reset_index().rename(columns={"exercise": "class"})
    tot = {"class": "All", "name": "", "train": N["n_train"], "val": N["n_val"], "test": N["n_test"], "total": N["usable_videos"]}
    parts = {"class": "Participants", "name": "", "train": N["p_train"], "val": N["p_val"], "test": N["p_test"], "total": N["participants_total"]}
    t1 = pd.concat([t1, pd.DataFrame([tot, parts])], ignore_index=True)
    R.write_table(t1, TAB / "table1_dataset_summary",
                  f"Dataset summary (MobiPhysio v{N['dataset_version']}). {N['inventory_videos']} videos in the "
                  f"inventory; {N['excluded_duplicates']} byte-identical duplicates with conflicting labels excluded; "
                  f"{N['usable_videos']} used. Participant-level split; no participant appears in two splits.",
                  nd=0, label="tab:dataset")

    # ---------------------------------------------------------------- main comparison (Table 2)
    rows = []
    for k in finals:
        r0 = runs[(k, seed0)]
        d = r0["pred"]
        y, p = d.y_true.values, d.y_pred.values
        ci = R.bootstrap_metric(d, nb, bseed, lambda sel: R.macro_f1_idx(y, p, sel))
        s = sm.loc[k]
        rows.append({"Model": DISPLAY[k], "key": k, "Seeds": int(s.n_seeds),
                     "Accuracy (mean ± SD)": R.pm(s.accuracy_mean, s.accuracy_sd),
                     "Macro precision (mean ± SD)": R.pm(s.macro_precision_mean, s.macro_precision_sd),
                     "Macro recall (mean ± SD)": R.pm(s.macro_recall_mean, s.macro_recall_sd),
                     "Macro F1 (mean ± SD)": R.pm(s.macro_f1_mean, s.macro_f1_sd),
                     f"Macro F1, seed {seed0}": r0["metrics"]["macro_f1"],
                     f"95% CI seed {seed0} (participant bootstrap)": f"[{ci[0]:.3f}, {ci[1]:.3f}]"})
        N[f"{k}.f1_mean"], N[f"{k}.f1_sd"] = float(s.macro_f1_mean), float(s.macro_f1_sd)
        N[f"{k}.acc_mean"], N[f"{k}.acc_sd"] = float(s.accuracy_mean), float(s.accuracy_sd)
        N[f"{k}.f1_seed0"], N[f"{k}.f1_seed0_ci"] = float(r0["metrics"]["macro_f1"]), [float(ci[0]), float(ci[1])]
        N[f"{k}.n_seeds"] = int(s.n_seeds)
        N[f"{k}.min_recall_mean"] = float(s.min_class_recall_mean)
        N[f"{k}.per_seed_f1"] = st[st.key == k].sort_values("seed").macro_f1.tolist()
        N[f"{k}.seeds"] = st[st.key == k].sort_values("seed").seed.tolist()
    t2 = pd.DataFrame(rows)
    R.write_table(t2.drop(columns="key"), TAB / "table2_main_comparison",
                  f"Main comparison on the participant-independent test set ({N['n_test']} videos, {N['p_test']} "
                  f"participants). Mean ± SD over training seeds (same split); the CI is a participant-cluster "
                  f"bootstrap ({nb} replicates) of the seed-{seed0} run only.", label="tab:main")
    st.assign(model=st.key.map(DISPLAY)).to_csv(TAB / "table2_per_seed_metrics.csv", index=False)

    # pairwise: MobiPhysio-Net vs each retained baseline (seed-averaged, participant bootstrap)
    pw = []
    for k in finals:
        if k == "mobiphysio_net":
            continue
        common = sorted(set(N["mobiphysio_net.seeds"]) & set(N[f"{k}.seeds"]))
        r = R.seed_averaged_paired_diff(runs, "mobiphysio_net", k, common, nb, bseed)
        pw.append({"Comparison": f"MobiPhysio-Net − {DISPLAY[k]}", "Seeds": ",".join(map(str, common)),
                   "Mean diff (macro F1)": r["mean_diff"], "SD over seeds": r["sd_diff"],
                   "Seeds favouring MobiPhysio-Net": f"{int(round(r['frac_seeds_a_better'] * len(common)))}/{len(common)}",
                   "95% CI (participant bootstrap of seed mean)": f"[{r['participant_boot_ci'][0]:.3f}, {r['participant_boot_ci'][1]:.3f}]"})
        N[f"vs_{k}"] = r
    R.write_table(pd.DataFrame(pw), TAB / "tableS1_pairwise_vs_baselines",
                  "MobiPhysio-Net minus each retained baseline: macro-F1 difference averaged over matched training "
                  "seeds; CI from a participant-cluster bootstrap applied jointly to all seeds.", label="tab:pairwise")

    # ---------------------------------------------------------------- ablation (Table 3)
    full = "mobiphysio_net"
    abl_rows, abl_seed_rows = [], []
    for k in [full] + abls:
        for _, r in st[st.key == k].sort_values("seed").iterrows():
            abl_seed_rows.append({"Variant": ABL_LABEL[k], "key": k, "Seed": int(r.seed), "Macro F1": r.macro_f1,
                                  "Accuracy": r.accuracy})
    for k in abls:
        common = sorted(set(N["mobiphysio_net.seeds"]) & set(st[st.key == k].seed))
        r = R.seed_averaged_paired_diff(runs, full, k, common, nb, bseed)
        s = sm.loc[k]
        abl_rows.append({"Variant": ABL_LABEL[k], "key": k, "Seeds": len(common),
                         "Macro F1 (mean ± SD)": R.pm(s.macro_f1_mean, s.macro_f1_sd),
                         "Accuracy (mean ± SD)": R.pm(s.accuracy_mean, s.accuracy_sd),
                         "Per-seed full − variant": ", ".join(f"{x:+.3f}" for x in r["per_seed_diff"]),
                         "Mean full − variant": r["mean_diff"], "SD": r["sd_diff"],
                         "Seeds full better": f"{int(round(r['frac_seeds_a_better'] * len(common)))}/{len(common)}",
                         "95% CI participants": f"[{r['participant_boot_ci'][0]:.3f}, {r['participant_boot_ci'][1]:.3f}]",
                         "95% CI seeds (t)": f"[{r['seed_t_ci'][0]:.3f}, {r['seed_t_ci'][1]:.3f}]"})
        N[f"abl_{k}"] = r
        N[f"{k}.f1_mean"], N[f"{k}.f1_sd"] = float(s.macro_f1_mean), float(s.macro_f1_sd)
        N[f"{k}.n_seeds"] = int(s.n_seeds)
    sfull = sm.loc[full]
    abl_rows.insert(0, {"Variant": ABL_LABEL[full], "key": full, "Seeds": int(sfull.n_seeds),
                        "Macro F1 (mean ± SD)": R.pm(sfull.macro_f1_mean, sfull.macro_f1_sd),
                        "Accuracy (mean ± SD)": R.pm(sfull.accuracy_mean, sfull.accuracy_sd)})
    t3 = pd.DataFrame(abl_rows)
    # per-class effect of removing DTW (seed-averaged recall difference)
    pcd = []
    for c in range(R.NC):
        dd = [runs[(full, s)]["metrics"]["per_class_f1"][c] - runs[("mobiphysio_net__no_dtw", s)]["metrics"]["per_class_f1"][c]
              for s in N["abl_mobiphysio_net__no_dtw"]["seeds"]]
        pcd.append({"Class": codes[c], "Name": names[c], "Mean F1 diff (full − no DTW)": float(np.mean(dd)),
                    "Min over seeds": float(np.min(dd)), "Max over seeds": float(np.max(dd))})
    # efficiency columns from the controlled benchmark
    eff_dir = ROOT / "results" / "efficiency"
    lat = pd.read_csv(eff_dir / "latency.csv") if (eff_dir / "latency.csv").exists() else pd.DataFrame()
    comps = pd.read_csv(eff_dir / "components.csv") if (eff_dir / "components.csv").exists() else pd.DataFrame()

    def lat_of(k, th, col="median_ms"):
        if lat.empty:
            return float("nan")
        x = lat[(lat.key == k) & (lat.threads == th)]
        return float(x[col].iloc[0]) if len(x) else float("nan")
    t3["Median latency 1 thread (ms)"] = [lat_of(k, 1) for k in t3.key]
    t3["Median latency 4 threads (ms)"] = [lat_of(k, 4) for k in t3.key]
    R.write_table(t3.drop(columns="key"), TAB / "table3_ablation",
                  "MobiPhysio-Net ablations on identical test videos. Differences are full − variant in macro F1; "
                  "'CI participants' = participant-cluster bootstrap of the seed-averaged difference (test-sample "
                  "uncertainty); 'CI seeds (t)' = t-interval over per-seed differences (training-seed variability). "
                  "Latency from the controlled benchmark (Table 5).", label="tab:ablation")
    pd.DataFrame(abl_seed_rows).drop(columns="key").to_csv(TAB / "table3_ablation_per_seed.csv", index=False)
    R.write_table(pd.DataFrame(pcd), TAB / "tableS2_dtw_per_class_effect",
                  "Per-class F1 change from adding the DTW branch (full − no-DTW), averaged over matched seeds.",
                  label="tab:dtwclass")
    N["dtw_class_effect"] = pcd

    # ---------------------------------------------------------------- per-class (Table 4)
    m0 = runs[(full, seed0)]["metrics"]
    pcm = [np.mean([runs[(full, s)]["metrics"]["per_class_f1"][c] for s in N["mobiphysio_net.seeds"]]) for c in range(R.NC)]
    rows4 = []
    for c in range(R.NC):
        row = {"Class": codes[c], "Name": names[c], f"Precision (seed {seed0})": m0["per_class_precision"][c],
               f"Recall (seed {seed0})": m0["per_class_recall"][c], f"F1 (seed {seed0})": m0["per_class_f1"][c],
               "F1 mean over seeds": pcm[c], "Support": int(m0["support"][c])}
        for k in finals:
            if k != full:
                row[f"F1 {DISPLAY[k]} (mean)"] = np.mean([runs[(k, s)]["metrics"]["per_class_f1"][c] for s in N[f"{k}.seeds"]])
        rows4.append(row)
    t4 = pd.DataFrame(rows4)
    R.write_table(t4, TAB / "table4_per_class",
                  f"Per-class results for MobiPhysio-Net (precision/recall/F1 for the seed-{seed0} run; F1 also "
                  "averaged over seeds) and seed-averaged F1 of the retained baselines.", label="tab:perclass")
    order = np.argsort(pcm)
    N["hardest_classes"] = [(codes[i], names[i], float(pcm[i])) for i in order[:3]]
    N["easiest_classes"] = [(codes[i], names[i], float(pcm[i])) for i in order[::-1][:3]]
    cm = m0["confusion"]
    N["confusion_seed0"] = cm.tolist()
    N["e05_to_e01"], N["e05_total"] = int(cm[4, 0]), int(cm[4].sum())
    N["e01_to_e05"], N["e01_total"] = int(cm[0, 4]), int(cm[0].sum())
    # pooled across seeds and all retained models: share of E05 predicted as E01
    pooled = {k: [runs[(k, s)]["metrics"]["confusion"] for s in N[f"{k}.seeds"]] for k in finals}
    N["e05_as_e01_rate_by_model"] = {k: float(np.mean([c[4, 0] / c[4].sum() for c in v])) for k, v in pooled.items()}
    off = np.array(cm, float)
    np.fill_diagonal(off, -1)
    i, j = np.unravel_index(off.argmax(), off.shape)
    N["max_offdiag_seed0"] = [codes[i], codes[j], int(cm[i, j])]
    # lying (E07-E09) <-> standing (E01-E06) confusions, pooled over retained models and seeds
    allc = [c for v in pooled.values() for c in v]
    N["lying_standing_cross"] = {"lying_as_standing": int(sum(c[6:, :6].sum() for c in allc)),
                                 "lying_total": int(sum(c[6:].sum() for c in allc)),
                                 "standing_as_lying": int(sum(c[:6, 6:].sum() for c in allc)),
                                 "standing_total": int(sum(c[:6].sum() for c in allc))}
    # superseded v1 run (duplicate pair included): seed-mean macro F1 of the retained models
    v1 = ROOT / "experiments" / "full_v1_superseded" / "registry.json"
    if v1.exists():
        reg1 = json.loads(v1.read_text())["seeds"]
        N["v1_f1_mean"] = {k: float(np.mean([json.loads((v1.parent / Path(rel).relative_to("experiments/full") / "metrics.json").read_text())["macro_f1"]
                                             for rel in reg1[k].values()])) for k in finals if k in reg1}
        N["v1_n_seeds"] = {k: len(reg1[k]) for k in finals if k in reg1}
        N["v1_max_abs_change"] = max(abs(v - N[f"{k}.f1_mean"]) for k, v in N["v1_f1_mean"].items())

    # ---------------------------------------------------------------- efficiency (Table 5)
    tt = pd.read_csv(eff_dir / "training_time.csv") if (eff_dir / "training_time.csv").exists() else pd.DataFrame()
    raw = json.loads((eff_dir / "benchmark_raw.json").read_text()) if (eff_dir / "benchmark_raw.json").exists() else {}
    dec = json.loads((eff_dir / "decode.json").read_text()) if (eff_dir / "decode.json").exists() else {}
    rows5 = []
    for k in finals + abls:
        x1 = lat[(lat.key == k) & (lat.threads == 1)] if not lat.empty else pd.DataFrame()
        x4 = lat[(lat.key == k) & (lat.threads == 4)] if not lat.empty else pd.DataFrame()
        g = lambda x, c: float(x[c].iloc[0]) if len(x) and c in x and pd.notna(x[c].iloc[0]) else float("nan")  # noqa: E731
        trn = tt[tt.key == k].train_eval_wall_s if not tt.empty else pd.Series(dtype=float)
        rows5.append({"Model": DISPLAY.get(k, k) if k in finals else ABL_LABEL[k], "key": k,
                      "Median ms (1 thr)": g(x1, "median_ms"), "p95 ms (1 thr)": g(x1, "p95_ms"),
                      "Median ms (4 thr)": g(x4, "median_ms"), "p95 ms (4 thr)": g(x4, "p95_ms"),
                      "Trainable params": g(x1, "params_trainable") if k != "svm" else float("nan"),
                      "Frozen backbone params": g(x1, "params_frozen_backbone"),
                      "SVM stored values": g(x1, "model_values") if k == "svm" else float("nan"),
                      "Checkpoint MB": g(x1, "checkpoint_bytes") / 1e6,
                      "Peak RSS MB (1 thr)": g(x1, "peak_rss_mb"),
                      "Train+eval wall time per run (min, mean)": float(trn.mean() / 60) if len(trn) else float("nan")})
        for c in ["Median ms (1 thr)", "Median ms (4 thr)", "Peak RSS MB (1 thr)", "p95 ms (1 thr)"]:
            N[f"{k}.{c}"] = rows5[-1][c]
    t5 = pd.DataFrame(rows5)
    for c in ["Trainable params", "Frozen backbone params", "SVM stored values"]:
        t5[c] = t5[c].round().astype("Int64")
    N["decode_median_s"], N["decode_p95_s"] = dec.get("median_s"), dec.get("p95_s")
    N["baseline_rss_mb"] = raw.get("baseline_rss_mb")
    if not comps.empty:
        ccc = comps[(comps.key == full) & (comps.threads == 1)].set_index("component")
        N["full_components_1t"] = ccc.median_ms.to_dict()
        ccc4 = comps[(comps.key == full) & (comps.threads == 4)].set_index("component")
        N["full_components_4t"] = ccc4.median_ms.to_dict()
    envj = json.loads((eff_dir / "environment.json").read_text()) if (eff_dir / "environment.json").exists() else {}
    N["hw"] = f"{envj.get('cpu_model', 'n/a')}, {envj.get('cpu_count', 'n/a')} logical CPUs, {envj.get('ram_gb', 'n/a')} GB RAM, no GPU"
    R.write_table(t5.drop(columns="key"), TAB / "table5_efficiency",
                  "Inference cost under a controlled benchmark: cached 16×224×224 frames → class probabilities, batch "
                  "of one video, fresh process per model, warm-up excluded, "
                  f"{int(lat.n_timed.iloc[0]) if not lat.empty else 'n/a'} test videos; MP4 decoding excluded "
                  f"(measured separately: median {dec.get('median_s', float('nan')):.2f} s per video). Hardware: {N['hw']}. "
                  "Peak RSS includes the Python/PyTorch runtime "
                  f"(idle baseline {raw.get('baseline_rss_mb', float('nan')):.0f} MB). SVM: stored support-vector values, "
                  "not trainable parameters. Training time: wall time of one train+evaluate run (features cached).",
                  label="tab:efficiency", nd=1)

    # ---------------------------------------------------------------- reproducibility (Table 6)
    man = runs[(full, seed0)]["manifest"]
    envr = man["environment"]
    t6 = pd.DataFrame([
        ("Dataset", f"MobiPhysio, Harvard Dataverse doi:10.7910/DVN/XSI0QN, version {N['dataset_version']} (CC0)"),
        ("Inputs", "16 uniformly spaced frames per video (PyAV accurate seek), letterboxed to 224×224 RGB"),
        ("Split", f"participant-level 70/10/20 by participants (grouped, class-balanced search, seed 42); "
                  f"{N['p_train']}/{N['p_val']}/{N['p_test']} participants"),
        ("Exclusions", f"{N['excluded_duplicates']} byte-identical videos with conflicting labels"),
        ("Spatial encoder", "timm vit_small_patch16_224.augreg_in21k_ft_in1k (frozen; variant D: last block + final norm trainable)"),
        ("Temporal encoder", "Linear 384→128 + LayerNorm, sinusoidal positions, 2-layer pre-norm Transformer (4 heads, FFN 256)"),
        ("DTW branch", "27 training-only references (3 DTW k-medoids/class), Sakoe-Chiba band 4, path-normalised, min aggregation, "
                       "training-fitted standardisation, no gradient"),
        ("Optimisation", "AdamW, one-cycle LR, class-weighted cross-entropy, early stopping on validation macro F1"),
        ("Checkpoint selection", "maximum validation macro F1; test set used once per run"),
        ("Seeds", ", ".join(map(str, sorted({s for (_, s) in runs})))),
        ("Uncertainty", f"participant-cluster bootstrap ({nb} replicates); SD and t-interval over seeds"),
        ("Software", f"Python {envr['python']}, PyTorch {envr['packages']['torch']}, timm {envr['packages']['timm']}, "
                     f"scikit-learn {envr['packages']['scikit-learn']}, PyAV {envr['packages']['av']}"),
        ("Hardware", f"{envr['cpu_model']}, {envr['cpu_count']} CPUs, {envr['ram_gb']} GB RAM, GPU: {envr['gpu']}"),
    ], columns=["Item", "Setting"])
    R.write_table(t6, TAB / "table6_reproducibility", "Implementation and reproducibility details.", label="tab:repro")

    # ---------------------------------------------------------------- supplementary: historical BiLSTM
    if hist:
        hs = []
        for k in hist:
            s = sm.loc[k]
            hs.append({"Model (historical, outside final scope)": DISPLAY[k], "Seeds": int(s.n_seeds),
                       "Per-seed macro F1": ", ".join(f"{x:.3f}" for x in st[st.key == k].sort_values("seed").macro_f1),
                       "Macro F1 (mean ± SD)": R.pm(s.macro_f1_mean, s.macro_f1_sd),
                       "Accuracy (mean ± SD)": R.pm(s.accuracy_mean, s.accuracy_sd)})
            N[f"{k}.f1_mean"], N[f"{k}.f1_sd"] = float(s.macro_f1_mean), float(s.macro_f1_sd)
            N[f"{k}.per_seed_f1"] = st[st.key == k].sort_values("seed").macro_f1.tolist()
        R.write_table(pd.DataFrame(hs), TAB / "tableS3_historical_bilstm",
                      "Supplementary: BiLSTM baseline trained with the same split, seeds and test videos but removed from "
                      "the final comparison scope by a project scope decision (decision log D4; no technical reason "
                      "claimed). Reported for transparency.", label="tab:bilstm")

    # ---------------------------------------------------------------- multi-criteria (Table S4)
    mc_tab = pd.DataFrame({"key": finals,
                           "macro_f1_mean": [N[f"{k}.f1_mean"] for k in finals],
                           "min_class_recall_mean": [N[f"{k}.min_recall_mean"] for k in finals],
                           "macro_f1_sd": [N[f"{k}.f1_sd"] for k in finals],
                           "latency_1t_ms": [N.get(f"{k}.Median ms (1 thr)", float("nan")) for k in finals],
                           "peak_rss_mb": [N.get(f"{k}.Peak RSS MB (1 thr)", float("nan")) for k in finals]})
    if mc_tab.drop(columns="key").notna().all().all():
        mc = R.multi_criteria(mc_tab)
        sc = mc["scores"].copy()
        sc.insert(0, "Model", sc.key.map(DISPLAY))
        raw_c = mc_tab.copy()
        raw_c.insert(0, "Model", raw_c.key.map(DISPLAY))
        R.write_table(raw_c.drop(columns="key"), TAB / "tableS4a_multicriteria_inputs",
                      "Multi-criteria inputs (raw, with units): " + "; ".join(f"{c} = {u}" for c, _, u in R.CRITERIA),
                      label="tab:mcinputs", nd=3)
        R.write_table(sc.drop(columns="key"), TAB / "tableS4b_multicriteria_scores",
                      "Composite scores (min–max normalised criteria, higher = better) under each weighting scheme. "
                      "Weights (F1, worst-class recall, seed SD, latency, memory): " +
                      "; ".join(f"{n}: {w}" for n, w in R.WEIGHT_SCHEMES.items()) +
                      ". The criteria and weights were specified after the results were known (see Statistical_Analysis.md).",
                      label="tab:mcscores", nd=3)
        N["mc_scores"] = mc["scores"].set_index("key").to_dict()
        N["mc_winner"] = {n: max(v, key=v.get) for n, v in N["mc_scores"].items()}
        N["mc_random_win_rate"] = mc["random_weight_win_rate"]

    # ---------------------------------------------------------------- Task B
    tb = json.loads((ROOT / "results" / "metrics" / "task_b_eaaq.json").read_text())
    N["taskb"] = tb
    tbm = pd.DataFrame(tb["metrics"])[["model", "n", "mae", "rmse", "r2", "spearman"]]
    R.write_table(tbm, TAB / "tableS5_eaaq_regression",
                  f"Exploratory EAAQ score regression (same participant split; {tb['n']['test']} test videos). "
                  "Baselines use only the training mean, the exercise code, or exercise + recording-variation code.",
                  label="tab:eaaq")

    # ---------------------------------------------------------------- figures
    make_figures(N, runs, st, sm, finals, abls, codes, names, lat, comps, seed0)
    DAT.mkdir(parents=True, exist_ok=True)
    (DAT / "final_numbers.json").write_text(json.dumps(N, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    from mobiphysio.reporting.manuscript import write_manuscript
    write_manuscript(N, MAN, finals, abls, hist)
    from mobiphysio.reporting.consistency import check
    n_fail = check(MAN, TAB, N, finals, abls, DISPLAY)
    print(f"consistency check: {n_fail} failed")
    write_indexes()
    print("paper assets written to", OUT)


def make_figures(N, runs, st, sm, finals, abls, codes, names, lat, comps, seed0):
    # Figure 1 — model comparison: macro F1 and accuracy, seed means ± SD with per-seed points
    order = sorted(finals, key=lambda k: N[f"{k}.f1_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (col, lab) in zip(axes, [("macro_f1", "Macro F1"), ("accuracy", "Accuracy")]):
        y = np.arange(len(order))
        means = [st[st.key == k][col].mean() for k in order]
        sds = [st[st.key == k][col].std(ddof=1) for k in order]
        ax.barh(y, means, color=[ACCENT if k == "mobiphysio_net" else NEUTRAL for k in order], height=0.62)
        ax.errorbar(means, y, xerr=sds, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for yi, k in zip(y, order):
            v = st[st.key == k][col].values
            ax.plot(v, np.full(len(v), yi), "o", ms=3.5, mfc="white", mec=INK, mew=0.8, zorder=3)
            ax.text(1.005, yi, f"{np.mean(v):.3f}", va="center", ha="left", fontsize=7.5, color=INK,
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Test {lab}")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(order)), [DISPLAY[k] for k in order])
    fig.suptitle(f"Figure 1. Retained models, participant-independent test set (n = {N['n_test']} videos); "
                 "bars: mean over seeds, error bars: SD, dots: seeds",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "figure1_model_comparison")

    # Figure 2 — MobiPhysio-Net confusion matrix (seed 0): counts + row-normalised recall
    cm = np.array(N["confusion_seed0"], float)
    rn = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, mat, title, fmt in [(axes[0], cm, "Counts", "{:.0f}"), (axes[1], rn, "Row-normalised (recall)", "{:.2f}")]:
        im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(9):
            for j in range(9):
                if mat[i, j] > 0:
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=6.5,
                            color="white" if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(9), codes, fontsize=7)
        ax.set_yticks(range(9), [f"{c} {n}" for c, n in zip(codes, names)], fontsize=6.5)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[1].set_yticklabels(codes)
    fig.colorbar(im, ax=axes, shrink=0.75, label="row-normalised recall")
    fig.suptitle(f"Figure 2. MobiPhysio-Net confusion matrix (seed {seed0}, {int(cm.sum())} test videos)",
                 x=0.01, ha="left", fontsize=9.5)
    save(fig, "figure2_mobiphysio_confusion")

    # Figure 3 — ablation: per-seed points + mean ± SD
    keys = ["mobiphysio_net"] + abls
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, k in enumerate(keys):
        v = st[st.key == k].sort_values("seed")
        m, s = v.macro_f1.mean(), v.macro_f1.std(ddof=1)
        ax.barh(i, m, color=ACCENT if k == "mobiphysio_net" else NEUTRAL, height=0.6)
        ax.errorbar(m, i, xerr=s, fmt="none", ecolor=INK, capsize=3, elinewidth=1)
        ax.plot(v.macro_f1, np.full(len(v), i), "o", ms=4, mfc="white", mec=INK, mew=0.8, zorder=3)
        ax.text(1.005, i, f"{m:.3f} (n={len(v)} seed{'s' if len(v) > 1 else ''})", va="center", fontsize=7.5,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(keys)), [ABL_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test macro F1 (mean ± SD over seeds; dots = seeds)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("Figure 3. MobiPhysio-Net ablations (same split and test videos)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "figure3_ablation")

    # Figure 4 — computational cost: (a) stacked component latency at 1 thread, (b) F1 vs latency
    if not comps.empty:
        ks = finals + abls
        lab = {k: (DISPLAY[k] if k in finals else ABL_LABEL[k].split(". ", 1)[1]) for k in ks}
        groups = {"backbone": "#2a78d6", "preprocess": "#a3a29c", "temporal": "#1baf7a", "DTW": "#eb6834", "head": "#52514e"}

        def grp(c):
            c = c.lower()
            if "dtw" in c:
                return "DTW"
            if "temporal" in c:
                return "temporal"
            if "backbone" in c or "cnn" in c:
                return "backbone"
            if "preprocess" in c:
                return "preprocess"
            return "head"
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
        ax = axes[0]
        c1 = comps[comps.threads == 1].copy()
        c1["g"] = c1.component.map(grp)
        y = np.arange(len(ks))
        left = np.zeros(len(ks))
        for g, col in groups.items():
            vals = np.array([c1[(c1.key == k) & (c1.g == g)].median_ms.sum() for k in ks])
            ax.barh(y, vals, left=left, color=col, height=0.6, label=g, edgecolor="white", linewidth=0.5)
            left += vals
        for yi, k in zip(y, ks):
            ax.text(left[yi] * 1.02, yi, f"{lat[(lat.key == k) & (lat.threads == 1)].median_ms.iloc[0]:.0f} ms",
                    va="center", fontsize=7)
        ax.set_yticks(y, [lab[k] for k in ks], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Median latency per video, 1 CPU thread (ms; sum of component medians)")
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title("(a) Where the time goes", loc="left", fontsize=8.5)
        ax = axes[1]
        for k in finals:
            x = lat[(lat.key == k) & (lat.threads == 1)].median_ms.iloc[0]
            ax.errorbar(x, N[f"{k}.f1_mean"], yerr=N[f"{k}.f1_sd"], fmt="o", ms=6,
                        color=ACCENT if k == "mobiphysio_net" else MUTED, capsize=2)
            off = {"lstm": (6, -11), "resnet50": (6, -9), "mobiphysio_net": (-8, 7)}.get(k, (5, 3))
            ax.annotate(DISPLAY[k], (x, N[f"{k}.f1_mean"]), xytext=off, textcoords="offset points", fontsize=7,
                        ha="right" if k == "mobiphysio_net" else "left")
        ax.set_xscale("log")
        ax.set_xlabel("Median latency, 1 thread (ms, log scale)")
        ax.set_ylabel("Macro F1 (mean ± SD over seeds)")
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title("(b) Accuracy–cost trade-off", loc="left", fontsize=8.5)
        fig.suptitle("Figure 4. Computational cost (cached frames → probabilities; MP4 decoding excluded)",
                     x=0.01, ha="left", fontsize=9.5)
        fig.tight_layout()
        save(fig, "figure4_computational_cost")

    # Figure 5 — per-class F1 (seed means) heatmap, retained models
    mat = np.array([[np.mean([runs[(k, s)]["metrics"]["per_class_f1"][c] for s in N[f"{k}.seeds"]]) for c in range(9)]
                    for k in finals])
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(9):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.55 else INK)
    ax.set_xticks(range(9), [f"{c}\n" + n.replace(" ", "\n", 1) for c, n in zip(codes, names)], fontsize=6.3)
    ax.set_yticks(range(len(finals)), [DISPLAY[k] for k in finals])
    ax.set_title("Figure 5. Per-class test F1 (mean over seeds)", loc="left", fontsize=9.5)
    save(fig, "figure5_per_class_f1")

    # Figure 6 — DTW effect per seed
    r = N["abl_mobiphysio_net__no_dtw"]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    xs = np.arange(len(r["seeds"]))
    ax.axhline(0, color=MUTED, lw=1)
    ax.bar(xs, r["per_seed_diff"], color=[ACCENT if d > 0 else "#eb6834" for d in r["per_seed_diff"]], width=0.55)
    for x, d in zip(xs, r["per_seed_diff"]):
        ax.text(x, d + (0.004 if d >= 0 else -0.004), f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=7.5)
    xm = len(xs) + 0.4
    lo, hi = r["participant_boot_ci"]
    ax.errorbar(xm, r["mean_diff"], yerr=[[r["mean_diff"] - lo], [hi - r["mean_diff"]]], fmt="D", color=INK, capsize=4)
    ax.text(xm + 0.15, hi, f"mean {r['mean_diff']:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]", fontsize=7, va="bottom")
    span = max(max(r["per_seed_diff"]), hi) - min(min(r["per_seed_diff"]), lo)
    ax.set_ylim(min(min(r["per_seed_diff"]), lo) - 0.2 * span, max(max(r["per_seed_diff"]), hi) + 0.25 * span)
    ax.set_xticks(list(xs) + [xm], [f"seed {s}" for s in r["seeds"]] + ["seed mean"])
    ax.set_xlim(-0.6, xm + 1.6)
    ax.set_ylabel("Macro F1: full − no-DTW")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("Figure 6. Effect of the DTW branch per training seed (CI: participant bootstrap of the seed mean)",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "figure6_dtw_effect_by_seed")


FIGS = [
    ("figure1_model_comparison", "Test macro F1 and accuracy of the retained models (mean ± SD over training seeds; dots are individual seeds).",
     "paper_assets/tables/table2_per_seed_metrics.csv"),
    ("figure2_mobiphysio_confusion", "MobiPhysio-Net confusion matrix for the primary seed: counts and row-normalised recall.",
     "paper_assets/data/final_numbers.json (confusion_seed0) ← experiments/<exp>/mobiphysio_net/<run>/test_predictions.csv"),
    ("figure3_ablation", "Ablation study: macro F1 per variant (mean ± SD over seeds; dots = seeds).",
     "paper_assets/tables/table3_ablation_per_seed.csv"),
    ("figure4_computational_cost", "(a) Median per-video latency decomposed by component (1 CPU thread); (b) macro F1 versus latency.",
     "results/efficiency/components.csv, results/efficiency/latency.csv, table2_per_seed_metrics.csv"),
    ("figure5_per_class_f1", "Per-class test F1 averaged over seeds for every retained model.",
     "experiments/<exp>/*/<run>/test_predictions.csv (via final_numbers.json)"),
    ("figure6_dtw_effect_by_seed", "Paired macro-F1 difference between the full model and the no-DTW variant for each seed, with the seed mean and its participant-bootstrap 95% CI.",
     "paper_assets/tables/table3_ablation.csv, final_numbers.json (abl_mobiphysio_net__no_dtw)"),
]


def write_indexes():
    lines = ["# Figure index", "", "All figures are generated by `scripts/make_paper_assets.py` from machine-readable results; "
             "no values were edited in image files.", "", f"Reproduce: `{CMD}`", "",
             "| File (PNG 300 dpi + PDF) | Caption | Source data |", "|---|---|---|"]
    for f, c, s in FIGS:
        lines.append(f"| `figures/{f}.png`, `figures/{f}.pdf` | {c} | `{s}` |")
    (OUT / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n")
    tl = ["# Table index", "", f"Reproduce: `{CMD}`. Each table exists as `.csv` (full precision), `.md` and `.tex`.", "",
          "| Table | Files |", "|---|---|"]
    for p in sorted(TAB.glob("*.csv")):
        stem = p.stem
        others = [x.name for x in TAB.glob(stem + ".*")]
        tl.append(f"| {stem} | {', '.join(sorted(others))} |")
    (OUT / "TABLE_INDEX.md").write_text("\n".join(tl) + "\n")


if __name__ == "__main__":
    main()
