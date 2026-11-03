"""Software / hardware environment capture for run manifests."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import sys

PACKAGES = ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "scipy",
            "opencv-python-headless", "av", "matplotlib", "PyYAML", "requests", "pytest",
            "huggingface_hub", "safetensors"]


def environment_info(path_for_disk: str = ".") -> dict:
    import torch
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_gb = None
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    du = shutil.disk_usage(path_for_disk)
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_model": cpu, "cpu_count": os.cpu_count(), "ram_gb": round(mem_gb, 1) if mem_gb else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)",
        "torch_threads": torch.get_num_threads(), "disk_free_gb": round(du.free / 1e9, 1),
        "packages": vers,
    }
