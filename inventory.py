"""Build and save the Dataverse metadata + inventory manifests."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pandas as pd

from .dataverse import DataverseClient, parse_inventory, summarize_metadata


def build_inventory(cfg: dict, out_dir: Path, client: DataverseClient | None = None) -> dict:
    client = client or DataverseClient(cfg["api_base"], timeout_s=cfg["download"]["timeout_s"],
                                       max_retries=cfg["download"]["max_retries"])
    doi = cfg["doi"]
    landing = client.resolve_doi(doi)
    ds = client.dataset(doi)
    meta = summarize_metadata(ds, doi, landing)
    inv = parse_inventory(ds, cfg["api_base"])

    # Duplicate content detection via published checksums.
    by_sum = collections.defaultdict(list)
    for r in inv:
        if r["checksum"]:
            by_sum[r["checksum"]].append(r["filename"])
    dups = {k: v for k, v in by_sum.items() if len(v) > 1}
    names = collections.Counter(r["filename"] for r in inv)
    meta.update({
        "n_videos_in_inventory": sum(1 for r in inv if (r["content_type"] or "").startswith("video/")),
        "total_size_bytes": int(sum(r["size_bytes"] or 0 for r in inv)),
        "content_types": dict(collections.Counter(r["content_type"] for r in inv)),
        "n_restricted": sum(r["restricted"] for r in inv),
        "duplicate_checksum_groups": dups,
        "duplicate_filenames": [n for n, c in names.items() if c > 1],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "dataverse_inventory.json").write_text(json.dumps(inv, indent=2))
    pd.DataFrame(inv).to_csv(out_dir / "dataverse_inventory.csv", index=False)
    return {"metadata": meta, "inventory": inv}


def load_inventory(out_dir: Path) -> list[dict]:
    path = out_dir / "dataverse_inventory.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `mobiphysio inventory` first")
    return json.loads(path.read_text())
