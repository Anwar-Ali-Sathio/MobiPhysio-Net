"""Machine-readable video manifest: inventory + parsed labels + preprocessing status
+ EAAQ linkage."""
from __future__ import annotations

import pandas as pd

from .labels import FilenameParser


def build_video_manifest(inventory: list[dict], parser: FilenameParser,
                         status: dict[str, dict] | None = None,
                         eaaq: pd.DataFrame | None = None,
                         class_order: list[str] | None = None) -> pd.DataFrame:
    status = status or {}
    class_order = class_order or sorted(parser.exercises)
    rows = []
    for r in inventory:
        if not (r["content_type"] or "").startswith("video/"):
            continue
        p = parser.parse(r["filename"])
        s = status.get(p["video_id"], {})
        rows.append({
            **p,
            "file_id": r["file_id"], "filename": r["filename"], "size_bytes": r["size_bytes"],
            "md5": r["checksum"],
            "label": class_order.index(p["exercise"]) if p["parse_ok"] else -1,
            "preprocess_status": s.get("status", "not_attempted"),
            "checksum_verified": bool(s.get("checksum_verified", False)),
            "decoder": s.get("decoder"), "fps": s.get("fps"), "width": s.get("width"),
            "height": s.get("height"), "n_frames": s.get("decoded_frames"),
            "n_valid_sampled": s.get("n_valid"), "padded": s.get("padded"),
        })
    df = pd.DataFrame(rows)
    if eaaq is not None and len(eaaq):
        named = eaaq[eaaq.video_name != ""].drop_duplicates("video_name", keep=False)
        m = named.set_index("video_name")
        df["eaaq_avg_score"] = df.video_id.map(m["avg_score"])
        df["eaaq_n_judges"] = df.video_id.map(m["n_judges"])
        df["eaaq_judge_sd"] = df.video_id.map(m["judge_sd"])
    # Exact-duplicate content (same MD5) under different filenames: if the copies carry
    # conflicting labels/metadata, the true identity is ambiguous -> exclude all copies.
    # (Identical-metadata copies would keep the first one.)
    dup_reason = {}
    meta_cols = ["exercise", "participant", "angle", "variation", "gender"]
    for md5, g in df[df.md5.notna()].groupby("md5"):
        if len(g) < 2:
            continue
        others = lambda v: ",".join(sorted(set(g.video_id) - {v}))
        if len(g[meta_cols].drop_duplicates()) > 1:
            for v in g.video_id:
                dup_reason[v] = f"duplicate content: MD5 {md5} shared with {others(v)} under conflicting labels"
        else:
            for v in sorted(g.video_id)[1:]:
                dup_reason[v] = f"duplicate content: identical copy of {sorted(g.video_id)[0]}"
    df["duplicate_of"] = df.video_id.map(dup_reason).fillna("")
    df["usable_task_a"] = df.parse_ok & (df.preprocess_status == "ok") & (df.duplicate_of == "")
    exclusion = []
    for _, r in df.iterrows():
        if r.duplicate_of:
            exclusion.append(r.duplicate_of)
        elif not r.parse_ok:
            exclusion.append(f"label parse: {r.parse_error}")
        elif r.preprocess_status != "ok":
            exclusion.append(f"preprocess: {r.preprocess_status}")
        else:
            exclusion.append("")
    df["exclusion_reason"] = exclusion
    return df.sort_values("video_id").reset_index(drop=True)


def find_annotation_orphans(eaaq: pd.DataFrame, video_ids: set[str]) -> list[str]:
    named = set(eaaq.loc[eaaq.video_name != "", "video_name"])
    return sorted(named - video_ids)
