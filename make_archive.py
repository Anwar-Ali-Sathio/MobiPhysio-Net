"""Create MobiPhysio-Net-Reproducible-Research.zip (code, configs, tests, docs, manifests,
logs, experiment records and results; NO raw videos, frame caches, features or checkpoints),
then re-open it, test every member's CRC and write its SHA-256 next to it."""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "MobiPhysio-Net-Reproducible-Research.zip"
EXCLUDE_SUFFIX = {".mp4", ".npz", ".npy", ".pt", ".part", ".pyc", ".zip"}
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def members():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts) or p.suffix in EXCLUDE_SUFFIX:
            continue
        yield p, Path("MobiPhysio-Net") / rel


def main(out_dir: Path) -> int:
    out = out_dir / NAME
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in members():
            z.write(src, arc)
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        names = z.namelist()
    if bad:
        print(f"CRC failure in {bad}")
        return 1
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (out_dir / (NAME + ".sha256")).write_text(f"{sha}  {NAME}\n")
    forbidden = [n for n in names if Path(n).suffix in EXCLUDE_SUFFIX]
    print(f"{out}\nmembers={len(names)} size={out.stat().st_size} bytes sha256={sha}\nforbidden={forbidden}")
    return 0 if not forbidden else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent))
