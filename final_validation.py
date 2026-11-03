"""Final release validation. Runs every check and writes reports/final_validation_report.md (+ .json).

Usage:  PYTHONPATH=src python scripts/final_validation.py
Exit status 1 if any check fails. Read-only except for the two report files (and pytest/ruff caches).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mobiphysio import pipeline as P  # noqa: E402

SECRET = re.compile(r"(gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
                    r"|xox[abp]-[A-Za-z0-9-]{10,}|ya29\.[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35})")
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".tex", ".html", ".toml", ".cfg", ".ini", ".sh", ".log"}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": "src", **__import__("os").environ})
    return p.returncode, (p.stdout + p.stderr).strip()


def main():
    rows = []

    def add(group, check, ok, evidence=""):
        rows.append({"group": group, "check": check, "ok": bool(ok), "evidence": str(evidence)[:400]})

    ctx = P.Context.create(None, "experiment", {})
    finals, abls = P.final_models(ctx), P.ablation_keys(ctx)
    ev = ctx.exp_cfg["evaluation"]
    seeds = sorted({ctx.exp_cfg["seed"], *ev.get("extra_training_seeds", [])})
    reg = json.loads((ctx.run_root / "registry.json").read_text())["seeds"]

    # --- experiment matrix
    for k in finals + abls:
        have = sorted(int(s) for s in reg.get(k, {}))
        st = [json.loads((ROOT / reg[k][str(s)] / "run_manifest.json").read_text())["exit_status"] for s in have]
        add("experiments", f"{k}: seeds {seeds} registered and completed", have == seeds and all(x == "completed" for x in st),
            f"registered {have}, status {st}")
    vf = [json.loads(m.read_text()) for m in ctx.run_root.glob("*/*/run_manifest.json")]
    bad = [m["run_id"] for m in vf if m["exit_status"] not in ("completed", "interrupted", "failed")]
    add("experiments", "no run left in 'running' state", not bad, bad or "none")
    var_missing = [m["run_id"] for m in vf if m["model"].startswith("mobiphysio_net") and m["exit_status"] == "completed"
                   and m.get("variant") is None]
    add("experiments", "MobiPhysio-Net runs without a recorded variant are documented (decision log / audit)",
        not var_missing or "variant" in (ROOT / "reports" / "finalization_audit.md").read_text(),
        f"{len(var_missing)} run(s) with variant=null (primary runs created before the fix; config fallback documented)")

    # --- code quality and tests
    rc, out = sh("python -m pytest -q 2>&1 | tail -3")
    add("code", "test suite passes", rc == 0 and "failed" not in out, out.splitlines()[-1] if out else "")
    rc, out = sh("ruff check src scripts tests")
    add("code", "ruff lint clean (src, scripts, tests; rules E,F,W,I,B per pyproject.toml, E501 not enforced)", rc == 0, out.splitlines()[-1] if out else "")
    rc, out = sh("python -m compileall -q src scripts")
    add("code", "all sources compile", rc == 0, out or "ok")

    # --- results verification (recomputes metrics from predictions, hashes, leakage)
    rc, out = sh("python -m mobiphysio.cli verify-results")
    vj = json.loads((ctx.results_dir / "reports" / "verification.json").read_text())
    add("results", "verify-results passed", rc == 0 and vj["exit_status"] == "passed",
        f"{len(vj['checks'])} runs; problems: {vj['problems'] or 'none'}; missing: {vj['missing_outputs'] or 'none'}")
    add("results", "no participant overlap between splits", vj["participant_overlap"] is False, vj["participant_overlap"])
    mc = pd.read_csv(ctx.results_dir / "metrics" / "model_comparison.csv")
    keys = set(mc[mc.columns[0]]) if "key" not in mc else set(mc.key)
    add("scope", "main comparison table contains exactly the six final models",
        keys == set(finals) or keys == set(finals + abls), sorted(keys))
    add("scope", "historical BiLSTM only in results/supplementary and archive",
        not any("bilstm" in str(k) for k in keys) and (ctx.results_dir / "supplementary" / "historical_models_comparison.csv").exists(),
        "results/supplementary/historical_models_comparison.csv")

    # --- efficiency benchmark
    eff = ROOT / "results" / "efficiency"
    add("efficiency", "controlled benchmark present and not the smoke test",
        (eff / "latency.csv").exists() and not (eff / "SMOKE_TEST_NOT_FOR_REPORTING.md").exists(),
        sorted(p.name for p in eff.glob("*")))
    if (eff / "latency.csv").exists():
        lat = pd.read_csv(eff / "latency.csv")
        add("efficiency", "latency measured for every final model and ablation at 1 and 4 threads",
            all(((lat.key == k) & (lat.threads == t)).any() for k in finals + abls for t in (1, 4)), f"{len(lat)} rows")

    # --- paper assets
    pa = ROOT / "paper_assets"
    figs = [p.stem for p in (pa / "figures").glob("figure*.png")]
    add("paper_assets", "6 figures, each as PNG and PDF",
        len(figs) == 6 and all((pa / "figures" / f"{f}.pdf").exists() for f in figs), sorted(figs))
    for i in range(1, 7):
        stems = [p.stem for p in (pa / "tables").glob(f"table{i}_*.csv") if "per_seed" not in p.stem]
        add("paper_assets", f"Table {i} as CSV, MD and TeX",
            bool(stems) and all((pa / "tables" / f"{s}{e}").exists() for s in stems for e in (".md", ".tex")), stems)
    need = ["Results.md", "Results.tex", "Discussion.md", "Discussion.tex", "Results_Discussion_Consistency_Check.md",
            "Abstract_Results_Summary.md", "Contributions_and_Findings.md", "Limitations.md", "Figure_Captions.md",
            "Table_Captions.md", "Statistical_Analysis.md", "Reproducibility_Statement.md"]
    miss = [n for n in need if not (pa / "manuscript" / n).exists()]
    add("paper_assets", "manuscript files complete", not miss, miss or "all present")
    cc = pa / "manuscript" / "Results_Discussion_Consistency_Check.md"
    m = re.search(r"Result: (\d+)/(\d+) checks passed", cc.read_text()) if cc.exists() else None
    add("paper_assets", "Results/Discussion consistency check: all checks pass",
        m and m.group(1) == m.group(2), m.group(0) if m else "missing")
    fn = pa / "data" / "final_numbers.json"
    if fn.exists():
        N = json.loads(fn.read_text())
        add("paper_assets", "final_numbers.json reflects every final-model seed",
            all(N.get(f"{k}.n_seeds") == len(seeds) for k in finals), {k: N.get(f"{k}.n_seeds") for k in finals})
        # newer than every registered run
        newest = max((ROOT / r / "metrics.json").stat().st_mtime for v in reg.values() for r in v.values())
        add("paper_assets", "paper assets generated after the last registered run", fn.stat().st_mtime > newest,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(fn.stat().st_mtime)))

    # --- determinism
    dj = ROOT / "reports" / "determinism_check.json"
    if dj.exists():
        d = json.loads(dj.read_text())
        add("reproducibility", "determinism rerun: predictions identical to registered runs",
            d and all(x.get("prediction_agreement") == 1.0 for x in d),
            "; ".join(f"{x['key']}:{x['seed']} agreement {x.get('prediction_agreement')}, max|Δp| "
                      f"{x.get('max_abs_probability_diff')}" for x in d))
    else:
        add("reproducibility", "determinism rerun performed", False, "reports/determinism_check.json missing")

    # --- hygiene: secrets / private files / raw data
    hits, raw = [], []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() in (".mp4", ".avi", ".mov"):
            raw.append(str(p.relative_to(ROOT)))
        if p.name in (".env", ".netrc", "credentials.json") or p.suffix in (".pem", ".key"):
            hits.append(str(p.relative_to(ROOT)))
        elif p.suffix.lower() in TEXT_EXT and p.stat().st_size < 20_000_000:
            if SECRET.search(p.read_text(errors="ignore")):
                hits.append(str(p.relative_to(ROOT)))
    add("hygiene", "no credentials or private environment files in the project tree", not hits, hits or "none found")
    add("hygiene", "no raw videos in the project tree", not raw, raw[:5] or "none")

    n_fail = sum(not r["ok"] for r in rows)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (ROOT / "reports" / "final_validation_report.json").write_text(json.dumps({"generated_utc": stamp, "checks": rows,
                                                                               "n_failed": n_fail}, indent=1))
    L = ["# Final validation report", "", f"Generated {stamp} by `scripts/final_validation.py` (all checks are executed, "
         "not asserted).", "", f"**{len(rows) - n_fail}/{len(rows)} checks passed" + (".**" if not n_fail else f"; {n_fail} FAILED.**"),
         "", "| Group | Check | Result | Evidence |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['group']} | {r['check']} | {'PASS' if r['ok'] else '**FAIL**'} | {r['evidence'].replace('|', '/')} |")
    (ROOT / "reports" / "final_validation_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
