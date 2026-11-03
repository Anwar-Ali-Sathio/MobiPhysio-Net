"""Build the final reproducible release and its delivery index.

Usage:  PYTHONPATH=src python scripts/make_release.py
Writes  dist/MobiPhysio-Net-Final-Reproducible-Release.zip (+ .sha256), checksums/release_SHA256SUMS.txt,
        checksums/release_manifest.json, DELIVERY_INDEX.md, reports/release_extraction_test.json.
The archive contains the repository (code, configs, data manifests, run records with per-video predictions,
results, paper assets, reports, logs, archive of historical runs) and no checkpoints, feature caches, raw videos
or credentials. After building, the archive is extracted into a fresh temporary directory and every file is
checked against the manifest; the test suite and verify-results are then run inside the extracted copy.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "MobiPhysio-Net-Final-Reproducible-Release"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", "dist"}
SKIP_SUFFIX = {".pt", ".pth", ".npz", ".npy", ".mp4", ".zip", ".pyc"}
SKIP_NAMES = {".env", ".netrc", "credentials.json", "DELIVERY_INDEX.md"}  # index describes the archive; shipped beside it


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or set(rel.parts) & SKIP_DIRS or p.suffix in SKIP_SUFFIX or p.name in SKIP_NAMES:
            continue
        if rel.parts[0] == "checksums" and rel.name.startswith("release_"):
            continue
        yield rel


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rels = list(files())
    man = [{"path": f"{NAME}/{r.as_posix()}", "bytes": (ROOT / r).stat().st_size, "sha256": sha(ROOT / r)} for r in rels]
    (ROOT / "checksums").mkdir(exist_ok=True)
    sums = "".join(f"{m['sha256']}  {m['path']}\n" for m in man)
    (ROOT / "checksums" / "release_SHA256SUMS.txt").write_text(sums)
    (ROOT / "checksums" / "release_manifest.json").write_text(json.dumps(
        {"release": NAME, "created_utc": stamp, "n_files": len(man), "total_bytes": sum(m["bytes"] for m in man),
         "excluded": "checkpoints (*.pt), feature/frame caches (*.npz/*.npy), raw videos, zip archives, caches, "
                     "credentials", "files": man}, indent=1))
    out = ROOT / "dist" / f"{NAME}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r in rels:
            z.write(ROOT / r, f"{NAME}/{r.as_posix()}")
        for extra in ("checksums/release_SHA256SUMS.txt", "checksums/release_manifest.json"):
            z.write(ROOT / extra, f"{NAME}/{extra}")
    zsha = sha(out)
    (out.parent / f"{out.name}.sha256").write_text(f"{zsha}  {out.name}\n")

    # extraction test in a fresh directory
    test = {"archive": str(out.relative_to(ROOT)), "archive_sha256": zsha, "archive_bytes": out.stat().st_size}
    with tempfile.TemporaryDirectory(prefix="release_test_") as td:
        with zipfile.ZipFile(out) as z:
            test["zip_testzip"] = z.testzip() is None
            z.extractall(td)
        base = Path(td) / NAME
        bad = [m["path"] for m in man if sha(Path(td) / m["path"]) != m["sha256"]]
        test["files_checked"], test["files_mismatched"] = len(man), bad
        env = {**os.environ, "PYTHONPATH": "src"}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=base, env=env,
                           capture_output=True, text=True)
        test["pytest"] = (r.stdout.strip().splitlines() or [""])[-1]
        test["pytest_ok"] = r.returncode == 0
        r = subprocess.run([sys.executable, "-m", "mobiphysio.cli", "verify-results"], cwd=base, env=env,
                           capture_output=True, text=True)
        vj = base / "results" / "reports" / "verification.json"
        v = json.loads(vj.read_text()) if vj.exists() else {}
        test["verify_results_ok"] = r.returncode == 0 and v.get("exit_status") == "passed"
        test["verify_results_runs"] = len(v.get("checks", []))
        test["verify_results_problems"] = v.get("problems")
        shutil.rmtree(base / "results" / "reports", ignore_errors=True)
    test["tested_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (ROOT / "reports" / "release_extraction_test.json").write_text(json.dumps(test, indent=1))
    write_index(stamp, man, test)
    print(json.dumps(test, indent=1))
    return 0 if test["zip_testzip"] and not bad and test["pytest_ok"] and test["verify_results_ok"] else 1


def write_index(stamp, man, test):
    def ex(p):
        return "yes" if (ROOT / p).exists() else "**missing**"
    items = [
        ("Release archive", f"dist/{NAME}.zip", "Everything below except checkpoints/caches; SHA-256 in the .sha256 file"),
        ("Release checksums", "checksums/release_SHA256SUMS.txt", "SHA-256 of every file in the archive"),
        ("Final report", "reports/final_report.md", "What was done, final results, limitations"),
        ("Final validation", "reports/final_validation_report.md", "Executed checks (tests, lint, verification, scope, assets)"),
        ("Determinism check", "reports/determinism_check.md", "Retrained runs compared with the registered runs"),
        ("Finalization audit", "reports/finalization_audit.md", "Project-state audit before finalization"),
        ("Decision log", "reports/experiment_decision_log.md", "Scope and protocol decisions (incl. BiLSTM removal, D4)"),
        ("Dataset audit", "results/reports/dataset_audit.md", "Inventory, checksums, exclusions, split leakage audit"),
        ("Main results", "results/metrics/model_comparison.csv", "Six final models"),
        ("Historical models", "results/supplementary/historical_models_comparison.csv", "BiLSTM, outside final scope"),
        ("Efficiency benchmark", "results/efficiency/latency.csv", "Controlled latency/memory benchmark"),
        ("Figures", "paper_assets/FIGURE_INDEX.md", "Figures 1–6 (PNG 300 dpi + PDF)"),
        ("Tables", "paper_assets/TABLE_INDEX.md", "Tables 1–6, S1–S5 (CSV/MD/TeX)"),
        ("Results section", "paper_assets/manuscript/Results.md", "Also .tex"),
        ("Discussion section", "paper_assets/manuscript/Discussion.md", "Also .tex"),
        ("Consistency check", "paper_assets/manuscript/Results_Discussion_Consistency_Check.md", "Text vs. computed results"),
        ("All numbers", "paper_assets/data/final_numbers.json", "Every number used in tables, figures and text"),
        ("Run registry", "experiments/full_v2/registry.json", "Run directory for every (model, seed)"),
        ("Recovery", "reports/next_steps_after_reconnection.md", "How to restore and resume"),
    ]
    L = ["# Delivery index — MobiPhysio-Net final release", "", f"Built {stamp}. Archive SHA-256 `{test['archive_sha256']}` "
         f"({test['archive_bytes']:,} bytes, {len(man):,} files).", "",
         f"Extraction test: zip integrity {'OK' if test['zip_testzip'] else 'FAILED'}; "
         f"{test['files_checked'] - len(test['files_mismatched'])}/{test['files_checked']} files match; pytest: "
         f"{test['pytest']}; verify-results: {'passed' if test['verify_results_ok'] else 'FAILED'} "
         f"({test['verify_results_runs']} runs).", "",
         "| Item | Path | Present | Contents |", "|---|---|---|---|"]
    L += [f"| {a} | `{p}` | {ex(p)} | {c} |" for a, p, c in items]
    L += ["", "Not included (size/type; regenerated by the documented commands): model checkpoints, feature and frame "
          "caches, raw MP4 videos (Harvard Dataverse doi:10.7910/DVN/XSI0QN, CC0)."]
    (ROOT / "DELIVERY_INDEX.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
