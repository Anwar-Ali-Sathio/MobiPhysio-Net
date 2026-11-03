"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(items: list[str] | None) -> dict:
    """Parse ``a.b.c=value`` strings into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def dataset_config() -> dict:
    return load_yaml(CONFIG_DIR / "dataset.yaml")


def experiment_config(name: str = "experiment") -> dict:
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def model_config(model: str) -> dict:
    path = CONFIG_DIR / "models" / f"{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config for {model!r} at {path}")
    return load_yaml(path)


def data_dir(override: str | None = None) -> Path:
    """Resolve the external data directory (CLI > env > dataset.yaml)."""
    raw = override or os.environ.get("MOBIPHYSIO_DATA_DIR") or dataset_config()["data_dir"]
    p = Path(raw)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_dir() -> Path:
    p = REPO_ROOT / dataset_config()["manifest_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = REPO_ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p
