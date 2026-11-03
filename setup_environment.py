"""Check Python, dependencies, CPU/RAM/GPU and disk before expensive jobs."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobiphysio.config import data_dir  # noqa: E402
from mobiphysio.reporting.environment import environment_info  # noqa: E402

info = environment_info(str(data_dir()))
missing = [k for k, v in info["packages"].items() if v is None and k not in ("pytest",)]
need_gb = 25  # frame cache (~9 GB) + ViT token cache (~9 GB) + features + transient downloads
info["checks"] = {"missing_packages": missing,
                  "disk_ok_for_full_run": shutil.disk_usage(data_dir()).free / 1e9 > need_gb,
                  "ram_ok": (info["ram_gb"] or 0) >= 12}
print(json.dumps(info, indent=2))
sys.exit(1 if missing else 0)
