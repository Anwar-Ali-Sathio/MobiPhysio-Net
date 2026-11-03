"""Controlled efficiency benchmark for all final models + ablations (+ historical, supplementary).

Usage:  PYTHONPATH=src python scripts/benchmark_efficiency.py [--videos 50] [--warmup 5] [--threads 4 1]
Writes: results/efficiency/{benchmark_raw.json, latency.csv, components.csv, training_time.csv, decode.json,
        environment.json}
Run on an otherwise idle machine; the script records the 1-minute load average before each model.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHILD = """
import json, sys
from pathlib import Path
from mobiphysio.evaluation.efficiency import benchmark_one
r = benchmark_one({key!r}, {threads}, {n}, {w}, Path({repo!r}), {exp!r})
print("JSONRESULT" + json.dumps(r))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--decode-videos", type=int, default=8)
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()
    import pandas as pd

    from mobiphysio import pipeline as P
    from mobiphysio.reporting.environment import environment_info
    ctx = P.Context.create(None, "experiment", {})
    out_dir = ROOT / "results" / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = args.keys or (P.final_models(ctx) + P.ablation_keys(ctx) + P.historical_models(ctx))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    raw = []
    # idle baseline: interpreter + torch + package import only
    base = subprocess.run([sys.executable, "-c", "import resource, torch, mobiphysio.pipeline; "
                           "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)"],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    baseline_rss = float(base.stdout.strip().splitlines()[-1])
    for th in args.threads:
        for k in keys:
            load = os.getloadavg()[0]
            code = CHILD.format(key=k, threads=th, n=args.videos, w=args.warmup, repo=str(ROOT), exp=ctx.exp_cfg["name"])
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
            line = [x for x in r.stdout.splitlines() if x.startswith("JSONRESULT")]
            if not line:
                raw.append({"key": k, "threads": th, "error": r.stderr[-2000:]})
                print(k, th, "FAILED", r.stderr[-500:])
                continue
            res = json.loads(line[0][len("JSONRESULT"):])
            res["loadavg_1min_before"] = load
            raw.append(res)
            print(f"{k:35s} threads={th} median={res['median_ms']:8.1f} ms  p95={res['p95_ms']:8.1f} ms  "
                  f"peakRSS={res['peak_rss_mb']:7.0f} MB  load={load:.2f}", flush=True)
    rows, comps = [], []
    for r in raw:
        if "error" in r:
            continue
        rows.append({k: r.get(k) for k in ["key", "threads", "n_timed", "median_ms", "p95_ms", "mean_ms", "peak_rss_mb",
                                           "rss_before_model_mb", "model_load_s", "params_trainable",
                                           "params_frozen_backbone", "model_values", "checkpoint_bytes",
                                           "load_time_reference_encoding_s", "loadavg_1min_before",
                                           "refit_reproduces_saved_predictions"]})
        for c, v in r["components"].items():
            comps.append({"key": r["key"], "threads": r["threads"], "component": c, **v})
    pd.DataFrame(rows).to_csv(out_dir / "latency.csv", index=False)
    pd.DataFrame(comps).to_csv(out_dir / "components.csv", index=False)
    # training wall time per run from run manifests
    reg = json.loads((ctx.run_root / "registry.json").read_text())
    tt = []
    for k, seeds in reg["seeds"].items():
        for s, rel in seeds.items():
            m = json.loads((ROOT / rel / "run_manifest.json").read_text())
            a = datetime.strptime(m["started_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            b = datetime.strptime(m["ended_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            tt.append({"key": k, "seed": int(s), "train_eval_wall_s": (b - a).total_seconds(),
                       "best_epoch": m.get("best_epoch"),
                       "note": "includes data loading, all epochs, validation, test prediction and in-run latency timing; "
                               "excludes one-off backbone feature extraction"})
    pd.DataFrame(tt).to_csv(out_dir / "training_time.csv", index=False)
    # MP4 decode benchmark on test videos
    inv = {Path(r["filename"]).stem: r for r in json.loads((ctx.manifest_dir / "dataverse_inventory.json").read_text())}
    sp = pd.read_csv(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv")
    test_ids = sorted(sp[sp.split == "test"].video_id)[:: max(1, (sp.split == "test").sum() // args.decode_videos)][: args.decode_videos]
    from mobiphysio.evaluation.efficiency import decode_benchmark
    dec = decode_benchmark([inv[v] for v in test_ids], ctx.data_dir / "decode_bench_tmp")
    (out_dir / "decode.json").write_text(json.dumps(dec, indent=2))
    (out_dir / "benchmark_raw.json").write_text(json.dumps({"baseline_rss_mb": baseline_rss, "runs": raw}, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(environment_info(str(ctx.data_dir)), indent=2))
    print("decode median s:", dec["median_s"], "baseline RSS MB:", baseline_rss)


if __name__ == "__main__":
    main()
