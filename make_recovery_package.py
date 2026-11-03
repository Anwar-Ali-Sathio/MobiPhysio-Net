"""Build the recovery package: staging copy -> manifest + SHA256SUMS -> zip -> 14 MB parts.

Usage: PYTHONPATH=src python scripts/make_recovery_package.py [--out DIR]
Package 1 (critical): repository (code, configs, manifests, run records, predictions, results, reports,
logs) + checkpoints of registered full_v2 runs except the 139 MB fine-tuned-ViT ones + decode status
records + the two small official dataset files (CC0).
Package 2 (large): fine-tuned-ViT checkpoints of registered full_v2 runs + vit_s16 / svm feature caches.
Only files of COMPLETED runs are copied (checkpoints are written once, at the end of a run).
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PART = 14 * 1024 * 1024
SKIP_DIRS = {"__pycache__", ".pytest_cache", "dist"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def category(rel):
    r = rel.replace("\\", "/")
    for pre, c in [("data_dir/checkpoints", "checkpoint"), ("data_dir/features", "feature_cache"),
                   ("data_dir/status", "decode_provenance"), ("data_dir/raw", "dataset_documentation"),
                   ("MobiPhysio-Net/src", "source"), ("MobiPhysio-Net/scripts", "source"), ("MobiPhysio-Net/tests", "tests"),
                   ("MobiPhysio-Net/configs", "config"), ("MobiPhysio-Net/data", "data_manifest"),
                   ("MobiPhysio-Net/experiments", "run_record"), ("MobiPhysio-Net/logs", "log"),
                   ("MobiPhysio-Net/results", "results"), ("MobiPhysio-Net/paper_assets", "paper_assets"),
                   ("MobiPhysio-Net/reports", "report"), ("MobiPhysio-Net/archive", "historical_archive"),
                   ("MobiPhysio-Net/checksums", "checksum"), ("MobiPhysio-Net/docs", "docs")]:
        if r.startswith(pre):
            return c
    return "project_file"


def build(stage: Path, files: dict, ts: str, label: str, out: Path, compress=True):
    """files: {relative path in archive: source path}"""
    manifest = []
    for rel, src in sorted(files.items()):
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        s1, s2 = sha(src), sha(dst)
        run_id = next((p for p in Path(rel).parts if p.count("-") >= 2 and "T" in p), None)
        manifest.append({"path": rel, "bytes": dst.stat().st_size, "sha256": s2, "source_sha256_match": s1 == s2,
                         "category": category(rel), "run_id": run_id, "source": str(src),
                         "required_for_recovery": category(rel) not in ("historical_archive",)})
    bad = [m["path"] for m in manifest if not m["source_sha256_match"]]
    mdir = stage / "MobiPhysio-Net" / "checksums"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{label}_manifest.json").write_text(json.dumps({"created_utc": ts, "files": manifest,
                                                             "changed_during_copy": bad}, indent=1))
    (mdir / f"{label}_SHA256SUMS.txt").write_text("".join(f"{m['sha256']}  {m['path']}\n" for m in manifest))
    name = f"MobiPhysio-Net-Recovery-{ts[:4]}{ts[5:7]}{ts[8:10]}-{ts[11:13]}{ts[14:16]}{'' if label == 'recovery' else '-' + label}.zip"
    zpath = out / name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
    parts = []
    with open(zpath, "rb") as f:
        i = 0
        while True:
            b = f.read(PART)
            if not b:
                break
            pp = out / f"{name}.part{i:02d}"
            pp.write_bytes(b)
            parts.append({"file": pp.name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
            i += 1
    info = {"archive": name, "archive_bytes": zpath.stat().st_size, "archive_sha256": sha(zpath), "parts": parts,
            "n_files": len(manifest), "total_uncompressed_bytes": sum(m["bytes"] for m in manifest),
            "changed_during_copy": bad, "created_utc": ts,
            "reassemble": f"cat {name}.part* > {name} && sha256sum {name}"}
    (out / f"{name}.parts.json").write_text(json.dumps(info, indent=1))
    return info, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "backup_out"))
    args = ap.parse_args()
    from mobiphysio import pipeline as P
    ctx = P.Context.create(None, "experiment", {})
    data = ctx.data_dir
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = json.loads((ctx.run_root / "registry.json").read_text())
    ck_small, ck_large = {}, {}
    for k, seeds in reg["seeds"].items():
        for rel in seeds.values():
            m = json.loads((ROOT / rel / "run_manifest.json").read_text())
            assert m["exit_status"] == "completed"
            ck = (m.get("outputs") or {}).get("checkpoint")
            if ck and Path(ck["path"]).exists():
                p = Path(ck["path"])
                assert sha(p) == ck["sha256"], f"checkpoint hash mismatch {p}"
                dst = f"data_dir/checkpoints/{k}/{p.name}"
                (ck_large if p.stat().st_size > 50e6 else ck_small)[dst] = p
    files = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (set(rel.parts) & SKIP_DIRS):
            files[f"MobiPhysio-Net/{rel.as_posix()}"] = p
    files.update(ck_small)
    for p in (data / "status_T16_S224").glob("*.json"):
        files[f"data_dir/status_T16_S224/{p.name}"] = p
    for p in (data / "raw").glob("*"):
        files[f"data_dir/raw/{p.name}"] = p
    for p in (data / "features" / "full").glob("*.json"):
        files[f"data_dir/features/full/{p.name}"] = p
    stage1 = Path(str(out / "stage_recovery"))
    shutil.rmtree(stage1, ignore_errors=True)
    info1, man1 = build(stage1, files, ts, "recovery", out)
    large = dict(ck_large)
    for n in ["vit_s16.npz", "svm_descriptors.npz"]:
        large[f"data_dir/features/full/{n}"] = data / "features" / "full" / n
    stage2 = Path(str(out / "stage_large"))
    shutil.rmtree(stage2, ignore_errors=True)
    info2, man2 = build(stage2, large, ts, "large", out, compress=False)
    summary = {"created_utc": ts, "packages": [info1, info2],
               "not_backed_up": {
                   "frames cache (8.3 GB, 3,686 x 16 x 224 x 224 x 3 uint8)": "regenerate: mobiphysio preprocess (re-downloads 158.7 GB)",
                   "ViT token cache vit_s16_lastblock_in.f16.npy (8.9 GB)": "regenerate: mobiphysio extract-features",
                   "frames112.u8.npy (2.2 GB), resnet50.npz (242 MB)": "regenerate: mobiphysio extract-features",
                   "raw MP4 videos": "not stored anywhere (deleted after verified decoding); official source: doi:10.7910/DVN/XSI0QN",
                   "checkpoints of superseded v1 / pilot runs": "historical only; predictions/metrics are in the archive",
                   "in-progress run state": "not checkpointed mid-run (run-level resume only)"}}
    (out / "backup_summary.json").write_text(json.dumps(summary, indent=1))
    shutil.rmtree(stage1)
    shutil.rmtree(stage2)
    print(json.dumps({k: (v if k != "packages" else [{x: p[x] for x in ["archive", "archive_bytes", "n_files", "archive_sha256"]} | {"parts": len(p["parts"])} for p in v]) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
