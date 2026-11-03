"""Automated Results/Discussion consistency check.

Re-derives the qualitative claims from the written table CSVs (not from the generator's internal state) and checks
that every number in the manuscript text matches a computed value. Writes
``manuscript/Results_Discussion_Consistency_Check.md``; returns the number of failed checks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NUM = re.compile(r"(?<![A-Za-z\d])[-+−]?\d[\d,]*(?:\.\d+)?%?")
# structural constants that legitimately appear in the text without being results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "16", "27", "40", "100", "128", "224", "256",
             "384", "10,000", "0–100", "95%", "2,1", "50", "70", "10", "20"}
BANNED = [r"\bstate[- ]of[- ]the[- ]art\b", r"\boutperform(s|ed)? all\b", r"\bsuperior to all\b",
          r"\bsignificant(ly)?\b", r"\breal[- ]time (capable|performance)\b", r"\bclinically validated\b",
          r"\bbest of all (tested )?models\b"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]


def _flatten(x, out):
    if isinstance(x, dict):
        for v in x.values():
            _flatten(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flatten(v, out)
    elif isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool):
        out.append(float(x))
    elif isinstance(x, str):
        try:
            out.append(float(x))
        except ValueError:
            pass


def _forms(v):
    f = set()
    for nd in (0, 1, 2, 3):
        for x in (v, abs(v), v * 100, v / 1000):
            if np.isfinite(x):
                f.add(f"{x:.{nd}f}")
                f.add(f"{x:+.{nd}f}")
    if float(v).is_integer():
        f.add(f"{int(v):,}")
    return f


def _norm(tok):
    return tok.replace("−", "-").rstrip(",").rstrip("%").rstrip(".")


def check(out_dir: Path, tab_dir: Path, N: dict, finals: list[str], abls: list[str], display: dict) -> int:
    res = (out_dir / "Results.md").read_text()
    dis = (out_dir / "Discussion.md").read_text()
    text = res + "\n" + dis
    rows = []

    def add(claim, ok, evidence):
        rows.append((claim, bool(ok), evidence))

    # 1. ranking recomputed from the per-seed CSV
    st = pd.read_csv(tab_dir / "table2_per_seed_metrics.csv")
    st = st[st.key.isin(finals)]
    means = st.groupby("key").macro_f1.mean().sort_values(ascending=False)
    rank = list(means.index).index("mobiphysio_net") + 1
    stated = re.search(r"MobiPhysio-Net ranked (\w+)", res)
    add("Stated rank of MobiPhysio-Net equals the rank recomputed from table2_per_seed_metrics.csv",
        stated and stated.group(1) == ORD[rank - 1], f"text: {stated.group(1) if stated else 'missing'}; recomputed: "
        f"{ORD[rank - 1]} ({', '.join(f'{display[k]} {v:.3f}' for k, v in means.items())})")
    add("Seed counts per retained model are equal (complete experiment matrix)",
        st.groupby("key").seed.nunique().nunique() == 1, st.groupby("key").seed.nunique().to_dict())

    # 2. pairwise claims against the Table S1 CSV
    s1 = pd.read_csv(tab_dir / "tableS1_pairwise_vs_baselines.csv")
    m = re.search(r"CI excluded zero against ([^;.]+)", res)
    claimed = [x.strip() for x in m.group(1).split(",")] if m else []
    for _, r in s1.iterrows():
        b = r["Comparison"].split(" − ", 1)[1]
        lo, hi = map(float, r["95% CI (participant bootstrap of seed mean)"].strip("[]").split(","))
        nf, ns = map(int, r["Seeds favouring MobiPhysio-Net"].split("/"))
        robust = nf == ns and lo > 0
        add(f"Claim 'consistently better than {b}' made only if all seeds favour it and CI excludes 0",
            (b in claimed) == robust, f"seeds {nf}/{ns}, CI [{lo:.3f}, {hi:.3f}], claimed: {b in claimed}")

    # 3. ablation verdicts against the Table 3 CSV
    t3 = pd.read_csv(tab_dir / "table3_ablation.csv")
    for _, r in t3.dropna(subset=["Per-seed full − variant"]).iterrows():
        d = [float(x) for x in str(r["Per-seed full − variant"]).split(",")]
        lo, hi = map(float, r["95% CI participants"].strip("[]").split(","))
        verdict = ("consistent_benefit" if all(x > 0 for x in d) and lo > 0 else
                   "consistent_harm" if all(x < 0 for x in d) and hi < 0 else "inconclusive")
        v = r["Variant"]
        if v.startswith("B."):
            ok = ("not demonstrated a robust benefit" in res) == (verdict == "inconclusive")
            add("DTW-ablation wording matches recomputed verdict", ok, f"{verdict}; per seed {d}; CI [{lo}, {hi}]")
        if v.startswith("C."):
            ok = ("a consistent effect under our criterion" in res) == (verdict == "consistent_benefit")
            add("Temporal-ablation wording matches recomputed verdict", ok, f"{verdict}; per seed {d}; CI [{lo}, {hi}]")

    # 4. scope: BiLSTM absent from the main table, figure 1 and the ranking; mentioned only as out of scope
    t2 = pd.read_csv(tab_dir / "table2_main_comparison.csv")
    add("BiLSTM absent from Table 2 (main comparison)", not t2.Model.str.contains("BiLSTM").any(), list(t2.Model))
    bil = [s for s in re.split(r"(?<=[.!?])\s", text) if "BiLSTM" in s]
    add("Every paragraph mentioning BiLSTM marks it as outside the comparison scope",
        all(re.search(r"scope", p) for p in text.split("\n\n") if "BiLSTM" in p), f"{len(bil)} sentence(s)")
    add("Text does not claim MobiPhysio-Net beats all models tested", not re.search(
        r"(best|highest) (of|among) all (models )?tested", text), "pattern search")

    # 5. banned over-claims
    for pat in BANNED:
        hits = re.findall(pat, text, flags=re.I)
        add(f"No over-claim matching /{pat}/", not hits, hits[:3] if hits else "none")

    # 6. every number in the text matches a computed value
    vals = []
    _flatten(N, vals)
    for csv in tab_dir.glob("*.csv"):
        for v in pd.read_csv(csv).select_dtypes("number").to_numpy().ravel():
            vals.append(float(v))
    allowed = set(CONSTANTS)
    for v in vals:
        if np.isfinite(v):
            allowed |= _forms(v)
    unmatched = []
    for name, t in (("Results.md", res), ("Discussion.md", dis)):
        clean = re.sub(r"\b(Table|Figure|Tables|Figures) S?\d+[a-z]?( and \d+)?|E0\d|P\d\d|ViT-S/16|ICC\(2,1\)|"
                       r"16×224×224|112×112|\(\w?i{1,3}v?\)", " ", t)
        for tok in NUM.findall(clean):
            if _norm(tok) not in {_norm(a) for a in allowed}:
                unmatched.append(f"{name}: {tok}")
    add("Every number in Results/Discussion equals a value in final_numbers.json or a table CSV (or a documented "
        "constant)", not unmatched, unmatched[:20] if unmatched else f"{len(NUM.findall(text))} numbers checked")
    add("No 'nan' in the manuscript text", "nan" not in re.findall(r"\bnan\b", text), "")

    n_fail = sum(not ok for _, ok, _ in rows)
    lines = ["# Results/Discussion consistency check", "",
             "Generated automatically by `src/mobiphysio/reporting/consistency.py` (called from "
             "`scripts/make_paper_assets.py`). Claims are re-derived from the written table CSVs, not from the "
             "manuscript generator. A human must still read the text.", "",
             f"**Result: {len(rows) - n_fail}/{len(rows)} checks passed" + (".**" if not n_fail else f"; {n_fail} FAILED.**"),
             "", "| # | Check | Result | Evidence |", "|---|---|---|---|"]
    for i, (c, ok, ev) in enumerate(rows, 1):
        ev = json.dumps(ev, default=str) if not isinstance(ev, str) else ev
        lines.append(f"| {i} | {c} | {'PASS' if ok else '**FAIL**'} | {ev.replace('|', '/')} |")
    (out_dir / "Results_Discussion_Consistency_Check.md").write_text("\n".join(lines) + "\n")
    return n_fail
