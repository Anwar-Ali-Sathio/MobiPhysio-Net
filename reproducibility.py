"""Seeds, determinism and run identifiers."""
from __future__ import annotations

import os
import random
import time
import uuid

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    notes = []
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            notes.append("torch.use_deterministic_algorithms(True, warn_only=True)")
        except Exception as exc:  # pragma: no cover
            notes.append(f"deterministic algorithms unavailable: {exc}")
    notes.append("CPU float reductions with >1 thread may differ in the last bits between "
                 "runs/machines; SVC probability calibration uses internal CV (seeded).")
    return {"seed": seed, "determinism_notes": notes}


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
