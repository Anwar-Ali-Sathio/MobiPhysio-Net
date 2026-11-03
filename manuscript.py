"""Manuscript text generated from paper_assets/data/final_numbers.json.

All numbers are inserted from the computed results; qualitative wording is chosen by explicit rules
(e.g. a component is only described as beneficial if every seed agrees AND the participant-bootstrap CI
of the seed-averaged difference excludes zero). Authors must still read and verify the text.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..evaluation.comparison import DISPLAY

ABL = {"mobiphysio_net__no_dtw": "the no-DTW variant (B)",
       "mobiphysio_net__dtw_no_temporal": "the variant without the temporal encoder (C)",
       "mobiphysio_net__finetuned_vit": "the last-block fine-tuned ViT variant (D)"}


def f3(x):
    return f"{x:.3f}"


def sgn(x):
    return f"{x:+.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def robust(r):
    """Rule for crediting a component: same sign on every seed AND participant CI excludes 0."""
    lo, hi = r["participant_boot_ci"]
    all_pos = all(d > 0 for d in r["per_seed_diff"])
    all_neg = all(d < 0 for d in r["per_seed_diff"])
    if all_pos and lo > 0:
        return "consistent_benefit"
    if all_neg and hi < 0:
        return "consistent_harm"
    return "inconclusive"


def md_to_tex(md: str) -> str:
    out = []
    in_list = False
    for line in md.splitlines():
        s = line
        s = s.replace("\\", "\\textbackslash{}")
        for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#") if not s.startswith("#") else ("", ""), ("_", "\\_")]:
            if a:
                s = s.replace(a, b)
        s = s.replace("±", "$\\pm$").replace("−", "$-$").replace("–", "--").replace("≥", "$\\geq$").replace("×", "$\\times$")
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
        s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
        if s.startswith("### "):
            s = "\\subsubsection{" + s[4:] + "}"
        elif s.startswith("## "):
            s = "\\subsection{" + s[3:] + "}"
        elif s.startswith("# "):
            s = "\\section{" + s[2:] + "}"
        if s.startswith("- ") or s.startswith("* "):
            if not in_list:
                out.append("\\begin{itemize}")
                in_list = True
            s = "  \\item " + s[2:]
        elif in_list and s.strip() == "":
            out.append("\\end{itemize}")
            in_list = False
        out.append(s)
    if in_list:
        out.append("\\end{itemize}")
    return "\n".join(out) + "\n"


def write_manuscript(N: dict, out: Path, finals, abls, hist):
    out.mkdir(parents=True, exist_ok=True)
    full = "mobiphysio_net"
    k0 = N["primary_seed"]
    ranked = sorted(finals, key=lambda k: -N[f"{k}.f1_mean"])
    rank = ranked.index(full) + 1
    top = ranked[0]
    nd = N["abl_mobiphysio_net__no_dtw"]
    nt = N["abl_mobiphysio_net__dtw_no_temporal"]
    ft = N["abl_mobiphysio_net__finetuned_vit"]
    dtw_verdict, temp_verdict = robust(nd), robust(nt)
    comp1 = N.get("full_components_1t", {})
    vit_ms = next((v for c, v in comp1.items() if "backbone" in c.lower()), float("nan"))
    dtw_ms = next((v for c, v in comp1.items() if "dtw" in c.lower()), float("nan"))
    lat1, lat4 = N.get(f"{full}.Median ms (1 thr)"), N.get(f"{full}.Median ms (4 thr)")
    lat1_nodtw = N.get("mobiphysio_net__no_dtw.Median ms (1 thr)")
    hard, easy = N["hardest_classes"], N["easiest_classes"]
    tb = N["taskb"]
    tbm = {m["model"]: m for m in tb["metrics"]}
    best_tb = min(tb["metrics"], key=lambda m: m["mae"])
    video_tb = [m for m in tb["metrics"] if m["model"] in ("ridge_vit", "temporal_vit")]
    vs = {k: N[f"vs_{k}"] for k in finals if k != full}
    better_than = [k for k, r in vs.items() if robust(r) == "consistent_benefit"]
    worse_than = [k for k, r in vs.items() if robust(r) == "consistent_harm"]
    tied = [k for k in vs if k not in better_than + worse_than]
    sds = {k: N[f"{k}.f1_sd"] for k in finals}
    mcw = N.get("mc_winner", {})
    mcr = N.get("mc_random_win_rate", {})

    def name(k):
        return DISPLAY[k]

    ord_words = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
    e05 = N["e05_as_e01_rate_by_model"]
    e05_hi = [k for k in finals if e05[k] >= 0.2]
    e05_lo = [k for k in finals if e05[k] < 0.2]
    mo = N["max_offdiag_seed0"]
    ls = N["lying_standing_cross"]
    vit_share = vit_ms / lat1 if lat1 else float("nan")
    tb_video_best = min(m["mae"] for m in video_tb) if video_tb else float("nan")
    tb_base_best = min(m["mae"] for m in tb["metrics"] if m["model"] not in ("ridge_vit", "temporal_vit"))
    nodtw_faster = lat1_nodtw is not None and lat1 is not None and lat1_nodtw <= lat1

    # ===================================================================== Results
    R = []
    R.append("# Results\n")
    R.append("## Dataset and experimental protocol\n")
    R.append(
        f"The MobiPhysio inventory (version {N['dataset_version']}) contained {N['inventory_files']} files, of which "
        f"{N['inventory_videos']} were videos; all videos were downloaded, verified against their published MD5 "
        f"checksums and decoded. Two videos were byte-identical but carried different participant and recording-"
        f"variation labels; because the correct label could not be determined, both were excluded, leaving "
        f"{N['usable_videos']} videos from {N['participants_total']} participants (Table 1). Videos were split at the "
        f"participant level into training ({N['n_train']} videos, {N['p_train']} participants), validation "
        f"({N['n_val']} videos, {N['p_val']} participants) and test ({N['n_test']} videos, {N['p_test']} participants) "
        f"sets; no participant appeared in more than one set (overlap: {N['participant_overlap_pairs']}). Every model "
        f"was trained with training seeds {', '.join(map(str, N[full + '.seeds']))} on this single split, selected by "
        f"validation macro F1, and evaluated once on the same test videos. We report the mean ± standard deviation "
        f"(SD) over seeds, which reflects training stochasticity, and, separately, 95% confidence intervals (CIs) from a "
        f"participant-cluster bootstrap ({N['bootstrap_iterations']} replicates), which reflect uncertainty due to the "
        f"sampling of test participants. No external dataset was evaluated.\n")

    R.append("## Main classification performance\n")
    R.append(
        f"MobiPhysio-Net reached a test macro F1 of {f3(N[full + '.f1_mean'])} ± {f3(N[full + '.f1_sd'])} and an "
        f"accuracy of {f3(N[full + '.acc_mean'])} ± {f3(N[full + '.acc_sd'])} (mean ± SD over "
        f"{N[full + '.n_seeds']} seeds; Table 2, Figure 1). For the primary seed ({k0}) the macro F1 was "
        f"{f3(N[full + '.f1_seed0'])} (95% participant-bootstrap CI {ci(N[full + '.f1_seed0_ci'])}). Among the "
        f"{len(finals)} retained models, MobiPhysio-Net ranked {['first', 'second', 'third', 'fourth', 'fifth', 'sixth'][rank - 1]} "
        f"by mean macro F1" + ("." if rank == 1 else f"; the highest mean macro F1 was obtained by {name(top)} "
                               f"({f3(N[top + '.f1_mean'])} ± {f3(N[top + '.f1_sd'])}).") + "\n")
    lines = []
    for k in finals:
        if k == full:
            continue
        r = vs[k]
        lines.append(f"- versus {name(k)} ({f3(N[k + '.f1_mean'])} ± {f3(N[k + '.f1_sd'])}): mean difference "
                     f"{sgn(r['mean_diff'])} (MobiPhysio-Net better in {int(round(r['frac_seeds_a_better'] * len(r['seeds'])))}"
                     f" of {len(r['seeds'])} seeds; 95% participant-bootstrap CI of the seed-averaged difference "
                     f"{ci(r['participant_boot_ci'])}).")
    R.append("Seed-matched macro-F1 differences between MobiPhysio-Net and each retained baseline were (Table S1):\n")
    R.append("\n".join(lines) + "\n")
    summ = []
    if better_than:
        summ.append("the difference was positive for every seed and its participant-bootstrap CI excluded zero against "
                    + ", ".join(name(k) for k in better_than))
    if tied:
        summ.append("the evidence did not establish a difference against " + ", ".join(name(k) for k in tied)
                    + " (sign varied across seeds and/or the CI included zero)")
    if worse_than:
        summ.append("MobiPhysio-Net was consistently worse than " + ", ".join(name(k) for k in worse_than))
    R.append("Under the criterion used throughout this paper (same sign for every seed and a participant-bootstrap CI "
             "excluding zero), " + "; ".join(summ) + ". These comparisons are descriptive; no correction for multiple "
             "comparisons was applied, and with three seeds and twelve test participants the analysis has limited power.\n")
    if hist:
        h = hist[0]
        R.append(f"A bidirectional LSTM (BiLSTM) baseline was also trained during the study with the same split, seeds "
                 f"and test videos and was subsequently removed from the comparison scope by a project scope decision "
                 f"(documented in the decision log; no performance-based justification is claimed). Its macro F1 was "
                 f"{f3(N[h + '.f1_mean'])} ± {f3(N[h + '.f1_sd'])} (per seed: "
                 f"{', '.join(f3(x) for x in N[h + '.per_seed_f1'])}), {f3(abs(N[h + '.f1_mean'] - N[full + '.f1_mean']))} "
                 f"{'higher' if N[h + '.f1_mean'] > N[full + '.f1_mean'] else 'lower'} than the MobiPhysio-Net mean; it is reported in "
                 f"Table S3 for completeness, and the comparisons above should not be read as a ranking of all models "
                 f"that were tested.\n")

    R.append("## Per-class results and confusion analysis\n")
    R.append(
        f"Per-class performance varied widely (Table 4, Figure 5). For MobiPhysio-Net the lowest seed-averaged F1 "
        f"values were for {hard[0][0]} {hard[0][1]} ({f3(hard[0][2])}), {hard[1][0]} {hard[1][1]} ({f3(hard[1][2])}) and "
        f"{hard[2][0]} {hard[2][1]} ({f3(hard[2][2])}), and the highest for {easy[0][0]} {easy[0][1]} ({f3(easy[0][2])}), "
        f"{easy[1][0]} {easy[1][1]} ({f3(easy[1][2])}) and {easy[2][0]} {easy[2][1]} ({f3(easy[2][2])}). "
        + (f"The largest off-diagonal entry of the confusion matrix (Figure 2, seed {k0}) was Circumduction (E05) "
           "predicted as Abduction (E01): " if mo[:2] == ["E05", "E01"] else
           f"The largest off-diagonal entry of the confusion matrix (Figure 2, seed {k0}) was {mo[0]} predicted as "
           f"{mo[1]} ({mo[2]} videos). Circumduction (E05) was predicted as Abduction (E01) for ")
        + f"{N['e05_to_e01']} of {N['e05_total']} E05 test videos, while {N['e01_to_e05']} of "
        f"{N['e01_total']} E01 videos were predicted as E05. The seed-averaged fraction of E05 videos labelled E01 was "
        + ", ".join(f"{name(k)} {e05[k]:.2f}" for k in finals) + ". "
        f"Pooled over all retained models and seeds, {ls['lying_as_standing']} of {ls['lying_total']} predictions for "
        f"test videos of lying exercises (E07–E09) were a standing exercise (E01–E06), and {ls['standing_as_lying']} of "
        f"{ls['standing_total']} predictions for standing-exercise videos were a lying exercise.\n")

    R.append("## Ablation study\n")
    R.append(
        f"Table 3 and Figures 3 and 6 compare the full model (A) with three variants on identical test videos and "
        f"matched seeds. Removing the DTW branch (B) changed macro F1 by "
        f"{', '.join(sgn(d) for d in nd['per_seed_diff'])} for seeds {', '.join(map(str, nd['seeds']))} (full − variant); "
        f"the mean paired difference was {sgn(nd['mean_diff'])} (SD {f3(nd['sd_diff'])}; full model better in "
        f"{int(round(nd['frac_seeds_a_better'] * len(nd['seeds'])))} of {len(nd['seeds'])} seeds). The 95% participant-"
        f"bootstrap CI of the seed-averaged difference was {ci(nd['participant_boot_ci'])} and the t-interval across "
        f"seeds was {ci(nd['seed_t_ci'])}. ")
    if dtw_verdict == "consistent_benefit":
        R[-1] += "The DTW branch therefore produced a consistent improvement under our criterion.\n"
    elif dtw_verdict == "consistent_harm":
        R[-1] += "Adding the DTW branch was consistently detrimental under our criterion.\n"
    else:
        R[-1] += ("The direction of the effect was not consistent across seeds, and the DTW branch has therefore not "
                  "demonstrated a robust benefit in this study."
                  + (" In particular, the favourable difference observed for individual seeds should not be generalised.\n"
                     if any(d > 0 for d in nd["per_seed_diff"]) else "\n"))
    pc = sorted(N["dtw_class_effect"], key=lambda r: r["Mean F1 diff (full − no DTW)"])
    R.append(f"At the class level, the seed-averaged F1 change from adding DTW ranged from "
             f"{sgn(pc[0]['Mean F1 diff (full − no DTW)'])} ({pc[0]['Class']}) to "
             f"{sgn(pc[-1]['Mean F1 diff (full − no DTW)'])} ({pc[-1]['Class']}) (Table S2).\n")
    R.append(
        f"Removing the learned temporal encoder (C; DTW and mean pooling applied directly to ViT features) changed macro "
        f"F1 by full − variant = {', '.join(sgn(d) for d in nt['per_seed_diff'])} for seeds "
        f"{', '.join(map(str, nt['seeds']))} (mean {sgn(nt['mean_diff'])}, participant-"
        f"bootstrap CI {ci(nt['participant_boot_ci'])})"
        + (", a consistent effect under our criterion.\n" if temp_verdict == "consistent_benefit" else ".\n"))
    R.append(
        f"Training the last ViT block together with the rest of the model (D; the final transformer block and final "
        f"layer normalisation of ViT-S/16 trainable, all earlier blocks frozen) yielded a macro F1 of "
        f"{f3(N['mobiphysio_net__finetuned_vit.f1_mean'])} ± {f3(N['mobiphysio_net__finetuned_vit.f1_sd'])} "
        f"(difference to A: mean {sgn(ft['mean_diff'])}, CI {ci(ft['participant_boot_ci'])}). This variant was trained "
        f"for at most eight epochs because of the CPU budget, whereas the frozen variants used up to 40 epochs; the "
        f"comparison therefore reflects the constrained training budget rather than the potential of fine-tuning.\n")

    R.append("## Computational cost and resource usage\n")
    R.append(
        f"Latency was measured in a controlled benchmark in which each model ran in a fresh process on the same test "
        f"videos, from cached 16×224×224 frames to class probabilities, with a batch of one video (Table 5, Figure 4); "
        f"MP4 decoding, which is common to all models, was measured separately (median {N['decode_median_s']:.2f} s per "
        f"video). On a single CPU thread MobiPhysio-Net required a median of {lat1:.0f} ms per video ({lat4:.0f} ms "
        f"with four threads). Of this, the ViT forward pass over 16 frames accounted for {vit_ms:.0f} ms and the DTW "
        f"branch (27 reference sequences, including distance standardisation) for {dtw_ms:.1f} ms; the no-DTW variant "
        f"required {lat1_nodtw:.0f} ms. ")
    cheap = sorted(finals, key=lambda k: N.get(f"{k}.Median ms (1 thr)", 1e9))
    R[-1] += ("The fastest retained models were " + ", ".join(f"{name(k)} ({N[k + '.Median ms (1 thr)']:.0f} ms)"
                                                          for k in cheap[:3]) +
              ", while the models built on a frozen ResNet50 backbone required "
              + ", ".join(f"{N[k + '.Median ms (1 thr)']:.0f} ms ({name(k)})" for k in finals if k in ("lstm", "resnet50")) +
              f". Peak resident memory during inference was {N[full + '.Peak RSS MB (1 thr)']:.0f} MB for MobiPhysio-Net "
              f"(idle Python/PyTorch baseline {N['baseline_rss_mb']:.0f} MB). An earlier observation, made with latency "
              "timed inside the training runs on a shared machine, was that the no-DTW variant was slower than the full "
              "model; " + ("in the controlled benchmark the no-DTW variant was faster, as expected from removing a "
                           "component, so the earlier observation is attributed to timing noise.\n" if nodtw_faster else
                           "the controlled benchmark reproduced this ordering; its cause was not identified.\n"))

    R.append("## Robustness and reproducibility\n")
    R.append(
        "Seed-to-seed variability was substantial relative to the differences between the stronger models: the SD of "
        "macro F1 across seeds was " + ", ".join(f"{sds[k]:.3f} ({name(k)})" for k in finals) + ". " + (("In a superseded run that differed by the inclusion of the duplicated video pair (one copy in training, "
                 "one in validation; same test videos), the seed-mean macro F1 values were " +
                 ", ".join(f"{f3(N['v1_f1_mean'][k])} ({name(k)}, {N['v1_n_seeds'][k]} seed"
                           f"{'s' if N['v1_n_seeds'][k] > 1 else ''})" for k in finals if k in N.get('v1_f1_mean', {})) +
                 ", i.e. changes of up to " +
                 f3(N['v1_max_abs_change']) +
                 " relative to the reported run. ") if N.get("v1_f1_mean") else "") + "All reported metrics were recomputed from the saved per-video predictions, and the DTW reference bank "
        "was verified to contain training videos only.\n")
    if mcw:
        prim = "primary (accuracy-led, constrained deployment)"
        R.append(
            "In a multi-criteria assessment (Table S4; criteria: mean macro F1, lowest per-class recall, seed SD, "
            "single-thread latency and peak memory; min–max normalised), the highest composite score under the primary "
            f"weighting was obtained by {name(mcw[prim])}; under alternative weightings the top model was "
            + "; ".join(f"{name(w)} ({n})" for n, w in mcw.items() if n != prim) +
            ". Across 10,000 random weight vectors the top model was " +
            ", ".join(f"{name(k)} in {v * 100:.0f}%" for k, v in sorted(mcr.items(), key=lambda x: -x[1])) +
            " of cases. Because the criteria and weights were specified after the results were known, this assessment is "
            "descriptive rather than confirmatory.\n")

    R.append("## Exploratory analysis of exercise-quality scores\n")
    R.append(
        f"For {tb['n']['linked_usable']} videos an averaged physiotherapist EAAQ score (0–100) could be linked by "
        f"filename (inter-rater ICC(2,1) = {tb['inter_judge_icc2_1']:.3f}). On the {tb['n']['test']} test videos the "
        f"lowest mean absolute error was achieved by the {best_tb['model'].replace('_', ' ')} baseline "
        f"({best_tb['mae']:.2f} points), whereas the video-feature regressors reached "
        + ", ".join(f"{m['mae']:.2f} ({m['model'].replace('_', ' ')})" for m in video_tb) +
        f" and a constant training mean {tbm['train_mean']['mae']:.2f} (Table S5). "
        + ("The video models therefore did not outperform simple baselines, and this analysis provides no evidence that "
           "the representations assess exercise quality.\n" if tb_video_best >= tb_base_best else
           "The best video model had a lower error than the simple baselines; this exploratory result was not tested "
           "further.\n"))
    res_md = "\n".join(R)
    (out / "Results.md").write_text(res_md)
    (out / "Results.tex").write_text(md_to_tex(res_md))

    # ===================================================================== Discussion
    D = ["# Discussion\n"]
    D.append("## Overall performance\n")
    D.append(
        f"MobiPhysio-Net recognised the nine AROM exercises of unseen participants with a mean macro F1 of "
        f"{f3(N[full + '.f1_mean'])}, the " + ("highest" if rank == 1 else f"{ord_words[rank - 1]}-highest")
        + f" mean among the {len(finals)} retained models. "
        + ("It was consistently better than " + ", ".join(name(k) for k in better_than) + ". " if better_than else "")
        + ("Its difference to " + ", ".join(name(k) for k in tied) + " changed sign across seeds or had a participant-"
           "bootstrap interval that included zero, so an advantage over " + ("this model" if len(tied) == 1 else "these models")
           + " was not established. " if tied else "")
        + ("It was consistently worse than " + ", ".join(name(k) for k in worse_than) + ". " if worse_than else "")
        + ("The evidence therefore supports describing MobiPhysio-Net as competitive with, not superior to, the strongest "
           "retained baselines. " if tied or worse_than else "")
        + (f"The BiLSTM baseline outside the comparison scope reached a mean macro F1 of {f3(N[hist[0] + '.f1_mean'])} "
           f"(Table S3), {'above' if N[hist[0] + '.f1_mean'] > N[full + '.f1_mean'] else 'below'} that of MobiPhysio-Net, "
           "so the ranking above does not extend to all models tested in the study.\n" if hist else "\n"))
    D.append("## Contribution of DTW and of temporal modelling\n")
    if dtw_verdict == "inconclusive":
        D.append(
            f"The central hypothesis that DTW-based reference features add useful information was not supported. The "
            f"paired effect of the DTW branch averaged {sgn(nd['mean_diff'])} macro F1 and changed sign across seeds "
            f"(range {sgn(min(nd['per_seed_diff']))} to {sgn(max(nd['per_seed_diff']))}). Several explanations are "
            "possible and were not tested: the temporal Transformer may already capture the ordering information that "
            "DTW summarises; distances to 27 medoid references compress a nine-dimensional signal that the classifier "
            f"may not need; and 16 frames per clip (median duration {N['median_duration_s']:.0f} s) provide a coarse temporal resolution for "
            "alignment. A robust conclusion would require more seeds, additional participant splits and alternative "
            "reference-bank designs.\n")
    else:
        D.append(f"The DTW branch showed a {dtw_verdict.replace('_', ' ')} (mean {sgn(nd['mean_diff'])}).\n")
    if temp_verdict == "consistent_benefit":
        D.append(
            f"{'In contrast, t' if dtw_verdict != 'consistent_benefit' else 'T'}emporal modelling mattered: removing the learned temporal encoder lowered macro F1 by "
            f"{f3(nt['mean_diff'])} on average, with the same sign for every seed and a participant-bootstrap CI that "
            "excluded zero. "
            + ("The temporal encoder, not the DTW branch, is therefore the component whose removal consistently reduced "
               "performance in this study.\n" if dtw_verdict != "consistent_benefit" else "\n"))
    else:
        D.append(f"Removing the temporal encoder changed macro F1 by {sgn(nt['mean_diff'])} on average, but the effect "
                 "did not meet our consistency criterion.\n")
    D.append("## Class-level behaviour\n")
    D.append(
        "Circumduction (E05) recognised as Abduction (E01) was a frequent error (seed-averaged rate ≥ 0.20) for "
         + ", ".join(name(k) for k in e05_hi) + (" and less frequent for " + ", ".join(name(k) for k in e05_lo)
                                                  if e05_lo else "") + ". Both "
        "exercises are standing shoulder movements in which the arm is raised away from the body; with 16 frames per "
        "clip, the circular trajectory that distinguishes circumduction may be under-sampled, and a frame-level ViT "
        "representation may emphasise posture over trajectory. These are hypotheses rather than tested explanations. "
        f"The best-recognised classes were {easy[0][0]}, {easy[1][0]} and {easy[2][0]} (seed-averaged F1 "
        f"{easy[2][2]:.2f}–{easy[0][2]:.2f}); the lying exercises (E07–E09) differ strongly in body orientation from the "
        "standing exercises, which plausibly makes them easier to separate.\n")
    D.append("## Recognition is not assessment\n")
    D.append(
        "The task evaluated here is the identification of which exercise is being performed. It does not establish "
        "whether an exercise is performed correctly, safely or with adequate range of motion. "
        + ("The exploratory EAAQ " if tb_video_best >= tb_base_best else "")
        + ("analysis illustrates this gap: despite high inter-rater agreement, video-feature regressors did not outperform "
           "a per-exercise mean. " if tb_video_best >= tb_base_best else "") + "Any use for clinical feedback would require dedicated movement-quality models and "
        "validation against expert assessment in the target population.\n")
    D.append("## Computational cost and deployment\n")
    D.append(
        f"On a single CPU thread the full model needed about {lat1 / 1000:.1f} s per video, mostly for the "
        f"{'ViT forward pass (' + f'{vit_share * 100:.0f}' + '%)' if vit_share >= 0.5 else 'cost of components other than the ViT'}; "
        f"the DTW branch added {dtw_ms:.0f} ms. "
        + ("Per-clip processing therefore takes longer than one second on the tested hardware, and " if lat1 > 1000 else "")
        + "claims about mobile or real-time deployment would require measurement on the target devices. "
        + f"The handcrafted SVM ({N['svm.Median ms (1 thr)']:.0f} ms) and the from-scratch CNNs "
        + f"({N['cnn2d.Median ms (1 thr)']:.0f} and {N['cnn_lstm.Median ms (1 thr)']:.0f} ms) were "
        + ("much faster" if max(N['svm.Median ms (1 thr)'], N['cnn2d.Median ms (1 thr)'], N['cnn_lstm.Median ms (1 thr)']) < lat1 / 5
           else "faster" if max(N['svm.Median ms (1 thr)'], N['cnn2d.Median ms (1 thr)'], N['cnn_lstm.Median ms (1 thr)']) < lat1
           else "not uniformly faster") + "; the SVM's "
        + f"macro F1 ({N['svm.f1_mean']:.3f}) was {'higher' if N['svm.f1_mean'] > max(N['cnn2d.f1_mean'], N['cnn_lstm.f1_mean']) else 'not higher'} "
        + f"than that of the CNNs ({N['cnn2d.f1_mean']:.3f}, {N['cnn_lstm.f1_mean']:.3f}). The choice of model therefore depends on the "
        "deployment constraint: where accuracy dominates and a few seconds per clip are acceptable, the pre-trained "
        "temporal models (including MobiPhysio-Net) are preferable; where latency or memory dominates, the SVM offers a "
        + ("considerably cheaper alternative at lower accuracy." if N['svm.f1_mean'] < N[full + '.f1_mean'] else
           "considerably cheaper alternative.")
        + ((" The descriptive multi-criteria assessment (Table S4) reflects this trade-off: "
            + "; ".join(f"{name(w)} scored highest under the "
                        + ", ".join(n.split(' (')[0] for n, ww in mcw.items() if ww == w)
                        + (" weightings" if sum(ww == w for ww in mcw.values()) > 1 else " weighting")
                        for w in dict.fromkeys(mcw.values()))
            + f"; across random weightings MobiPhysio-Net was the top model in {mcr.get(full, 0) * 100:.0f}% of cases.")
           if mcw else "") + "\n")
    D.append("## Limitations\n")
    D.append(
        f"The study used a single dataset and a single participant-level split with {N['p_test']} test participants; "
        "split-to-split variance and external validity were not assessed, and the participants were filmed in a small "
        "number of shared recording environments. All experiments were CPU-constrained: the ViT and ResNet50 backbones "
        "were frozen, fine-tuning was limited to the last ViT block for at most eight epochs, and the from-scratch CNNs "
        "were trained at 112×112 pixels, which disadvantages them relative to models with pre-trained features. The "
        "baselines were not extensively tuned. " + f"{['One', 'Two', 'Three', 'Four', 'Five'][N[full + '.n_seeds'] - 1]} training seeds per configuration provide only a coarse estimate of "
        "training variability. The multi-criteria assessment was specified after the results were known. A BiLSTM "
        "baseline was removed from the comparison scope by a project decision and is reported only in the supplement.\n")
    D.append("## Future work\n")
    D.append(
        "Priorities are: (i) repeating the evaluation over several participant-level splits and more seeds; (ii) denser "
        "temporal sampling or pose-based trajectories to separate E01 from E05; (iii) alternative DTW designs (learned "
        "or differentiable alignment, more references, class-conditional distances) evaluated with the seed-paired "
        "protocol used here; (iv) full fine-tuning on GPU hardware; (v) latency measurement on target mobile devices; "
        "and (vi) movement-quality assessment with validated expert labels as a separate task.\n")
    dis_md = "\n".join(D)
    (out / "Discussion.md").write_text(dis_md)
    (out / "Discussion.tex").write_text(md_to_tex(dis_md))

    # ===================================================================== support files
    (out / "Abstract_Results_Summary.md").write_text(
        "# Abstract — results summary (draft)\n\n"
        f"On a participant-independent test set of {N['n_test']} videos from {N['p_test']} participants, MobiPhysio-Net "
        f"achieved a macro F1 of {f3(N[full + '.f1_mean'])} ± {f3(N[full + '.f1_sd'])} (mean ± SD, {N[full + '.n_seeds']} "
        f"seeds), ranking {ord_words[rank - 1]} of {len(finals)} retained models by mean macro F1; "
        + (("the difference to " + ", ".join(name(k) for k in tied) + " was not consistent across seeds and "
            "participants. ") if tied else "the differences to all retained baselines were consistent across seeds and participants. ")
        + f"Removing the temporal encoder changed macro F1 by {sgn(-nt['mean_diff'])} (same direction for every seed: "
        f"{'yes' if temp_verdict != 'inconclusive' else 'no'}), and removing the DTW branch by {sgn(-nd['mean_diff'])} "
        f"(same direction for every seed: {'yes' if dtw_verdict != 'inconclusive' else 'no'}). "
        f"Inference took a median of {lat1:.0f} ms per video on one CPU thread, "
        f"{'dominated by' if vit_share >= 0.5 else 'including'} the ViT backbone (DTW: {dtw_ms:.0f} ms). "
        + ("The most frequent error was confusion of circumduction with abduction. " if mo[:2] == ["E05", "E01"] else "")
        + ("Video features did not predict physiotherapist quality scores better than simple baselines.\n"
           if tb_video_best >= tb_base_best else "\n"))
    (out / "Contributions_and_Findings.md").write_text(
        "# Contributions and findings\n\n## Contributions (what was built and evaluated)\n\n"
        "- A reproducible pipeline for MobiPhysio: checksum-verified acquisition of all videos, participant-level split, "
        "duplicate detection, and per-run provenance records.\n"
        "- MobiPhysio-Net: frozen ViT-S/16 frame features, a temporal Transformer, and no-gradient DTW distances to a "
        "training-only reference bank, fused in an MLP classifier.\n"
        "- A seed-paired, participant-bootstrap evaluation protocol applied to six retained model families and three "
        "ablations, plus a controlled per-component latency benchmark.\n\n## Findings (evidence-proportional)\n\n"
        f"- MobiPhysio-Net: macro F1 {f3(N[full + '.f1_mean'])} ± {f3(N[full + '.f1_sd'])}, {ord_words[rank - 1]} of "
        f"{len(finals)} retained models; "
        + ("not demonstrably superior to " + ", ".join(name(k) for k in tied + worse_than) if tied or worse_than
           else "consistently better than every retained baseline") + ".\n"
        f"- Temporal encoder (full − no temporal, per seed {', '.join(sgn(d) for d in nt['per_seed_diff'])}): "
        f"{temp_verdict.replace('_', ' ')}.\n"
        f"- DTW branch (full − no DTW, per seed {', '.join(sgn(d) for d in nd['per_seed_diff'])}): "
        f"{dtw_verdict.replace('_', ' ')}.\n"
        f"- DTW branch cost: {dtw_ms:.0f} ms of {lat1:.0f} ms on one thread; ViT backbone: {vit_ms:.0f} ms.\n"
        "- A published duplicate video pair with conflicting labels was identified and excluded.\n"
        + ("- Exercise-quality (EAAQ) prediction from these video features did not beat simple baselines.\n"
           if tb_video_best >= tb_base_best else "- A video model had the lowest EAAQ error (exploratory).\n"))
    (out / "Limitations.md").write_text("# Limitations\n\n" + D[D.index("## Limitations\n") + 1])
    (out / "Figure_Captions.md").write_text(
        "# Figure captions\n\n"
        f"**Figure 1.** Test macro F1 (left) and accuracy (right) of the retained models on the participant-independent "
        f"test set ({N['n_test']} videos, {N['p_test']} participants). Bars: mean over training seeds; error bars: SD; "
        "dots: individual seeds. BiLSTM is not shown (removed from the comparison scope; see Table S3).\n\n"
        f"**Figure 2.** Confusion matrix of MobiPhysio-Net for training seed {k0} ({sum(map(sum, N['confusion_seed0']))} "
        "test videos). Left: counts; right: row-normalised recall. Rows: true class; columns: predicted class.\n\n"
        "**Figure 3.** Ablation of MobiPhysio-Net: macro F1 of the full model and of variants without DTW, without the "
        "temporal encoder, and with the last ViT block trained (mean ± SD over seeds; dots = seeds). Variant D was "
        "trained for at most eight epochs.\n\n"
        "**Figure 4.** Computational cost from a controlled benchmark (cached frames → probabilities, batch of one video, "
        "MP4 decoding excluded). (a) Median latency per video on one CPU thread, decomposed into components. (b) Mean "
        "macro F1 (± SD over seeds) versus single-thread median latency (log scale).\n\n"
        "**Figure 5.** Per-class test F1 averaged over training seeds for each retained model.\n\n"
        "**Figure 6.** Macro-F1 difference between the full model and the no-DTW variant for each training seed, and "
        "the seed mean with its 95% participant-cluster bootstrap CI.\n")
    (out / "Table_Captions.md").write_text(
        "# Table captions\n\n"
        "**Table 1.** Dataset summary: per-class video counts in the participant-level training, validation and test "
        "sets, and participants per set.\n\n"
        "**Table 2.** Main comparison of the retained models: accuracy, macro precision, macro recall and macro F1 "
        "(mean ± SD over training seeds) and the participant-bootstrap 95% CI of macro F1 for the primary seed.\n\n"
        "**Table 3.** Ablation study: per-seed and mean paired differences (full − variant), SD, number of seeds "
        "favouring the full model, participant-bootstrap and seed t-intervals, and single- and four-thread latency.\n\n"
        "**Table 4.** Per-class precision, recall, F1 and support for MobiPhysio-Net and seed-averaged F1 of the "
        "retained baselines.\n\n"
        "**Table 5.** Efficiency measurements: median and 95th-percentile latency, parameter counts, checkpoint size, "
        "peak memory, and training time, with measurement conditions.\n\n"
        "**Table 6.** Implementation and reproducibility details.\n\n"
        "**Table S1.** Seed-matched differences between MobiPhysio-Net and each retained baseline. **Table S2.** "
        "Per-class effect of the DTW branch. **Table S3.** Historical BiLSTM results (outside the comparison scope). "
        "**Table S4.** Multi-criteria inputs and composite scores. **Table S5.** Exploratory EAAQ regression.\n")
    (out / "Statistical_Analysis.md").write_text(
        "# Statistical analysis\n\n"
        "- **Unit of analysis.** Metrics are computed over test videos; uncertainty is attributed to participants, "
        "because videos of one participant (including simultaneous camera angles) are not independent.\n"
        f"- **Participant-cluster bootstrap.** Test participants ({N['p_test']}) are resampled with replacement "
        f"({N['bootstrap_iterations']} replicates, seed fixed); all videos of a sampled participant enter the replicate; "
        "percentile 95% intervals. For model or ablation differences the same resample is applied to both models and to "
        "every seed, and the statistic is the seed-averaged macro-F1 difference.\n"
        "- **Training-seed variability.** Each configuration was trained with the same seeds on the same split; we "
        "report mean ± SD (ddof = 1) and a t-interval over per-seed paired differences (df = n_seeds − 1). With three "
        "seeds this interval is wide.\n"
        "- **Decision rule for crediting a component or model difference.** Same sign for every seed AND a participant-"
        "bootstrap CI of the seed-averaged difference that excludes zero. This rule was adopted after a single-seed "
        "bootstrap was found to disagree in sign with the other seeds; it is conservative and is stated explicitly.\n"
        "- **No significance tests with multiplicity control** were performed; no p-values are reported, and no claim of "
        "statistical significance is made.\n"
        "- **Multi-criteria assessment.** Five criteria (mean macro F1, lowest per-class recall, seed SD, single-thread "
        "median latency, peak memory) are min–max normalised across the retained models (direction-adjusted) and "
        "combined with five weighting schemes and 10,000 Dirichlet(1) random weight vectors. The criteria and weights "
        "were defined after the results were known; the analysis is descriptive.\n"
        "- **Split variance not estimated.** A single participant split was used.\n")
    (out / "Reproducibility_Statement.md").write_text(
        "# Reproducibility statement\n\n"
        "Code, configuration, data manifests (inventory with checksums, split assignment, exclusions), per-run "
        "manifests (seed, configuration, environment, hashes), per-video test predictions, and the scripts that "
        "generate every table and figure are provided in the release archive. The MobiPhysio videos are not "
        "redistributed; they are available under CC0 from Harvard Dataverse (doi:10.7910/DVN/XSI0QN, version "
        f"{N['dataset_version']}) and are re-acquired and verified by `python -m mobiphysio.cli preprocess`. Model "
        "checkpoints and feature caches are not included because of their size; they are regenerated by the "
        "documented commands, and their SHA-256 hashes are recorded in the run manifests. All experiments ran on a "
        "CPU-only machine; exact bitwise reproduction of trained weights is not guaranteed across hardware because of "
        "non-deterministic floating-point reductions.\n")
    return {"rank": rank, "dtw_verdict": dtw_verdict, "temporal_verdict": temp_verdict}
