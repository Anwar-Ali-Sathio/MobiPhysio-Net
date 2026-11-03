"""Filename parsing and EAAQ annotation parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilenameParser:
    pattern: str
    exercises: dict
    angle_codes: dict
    variation_codes: dict

    @classmethod
    def from_config(cls, cfg: dict) -> "FilenameParser":
        f = cfg["filename"]
        return cls(f["pattern"], dict(cfg["exercises"]), dict(f["angle_codes"]),
                   dict(f["variation_codes"]))

    def parse(self, filename: str) -> dict:
        """Parse a filename. Returns ``parse_ok=False`` with an error, never guesses."""
        stem = Path(filename).stem
        m = re.match(self.pattern, stem)
        out = {"video_id": stem, "parse_ok": False, "parse_error": "", "parse_warning": "",
               "exercise": None, "exercise_name": None, "participant": None,
               "angle": None, "variation": None, "gender": None}
        if not m:
            out["parse_error"] = "filename does not match configured pattern"
            return out
        g = m.groupdict()
        # Errors invalidate the exercise label; warnings only flag undocumented metadata.
        errors, warnings = [], []
        if g.get("exercise") not in self.exercises:
            errors.append(f"unknown exercise code {g.get('exercise')}")
        if not g.get("participant"):
            errors.append("missing participant id")
        if g.get("angle") not in self.angle_codes:
            warnings.append(f"undocumented angle code {g.get('angle')}")
        if g.get("variation") not in self.variation_codes:
            warnings.append(f"undocumented variation code {g.get('variation')}")
        out.update({
            "exercise": g.get("exercise"),
            "exercise_name": self.exercises.get(g.get("exercise")),
            "participant": g.get("participant"),
            "angle": g.get("angle"),
            "variation": g.get("variation"),
            "gender": g.get("gender"),
        })
        out["parse_error"] = "; ".join(errors)
        out["parse_warning"] = "; ".join(warnings)
        out["parse_ok"] = not errors
        return out


def parse_eaaq_csv(path: str | Path, judges_per_video: int = 3) -> tuple[pd.DataFrame, dict]:
    """Parse ``Physiotherapist Exercise Marking .csv``.

    Layout (verified by inspection): two header rows, then blocks of rows - one per
    judge. The first row of a block carries the video name and the averaged score
    (0-100); following rows have an empty name column. Columns 2-5 are the four
    "Primary Factors" items, 6-8 the three "Control Factors" items, then the judge's
    total mark (0-7), that mark on a 0-100 scale, the average of the judges and the
    rounded average.

    Conservative linkage rule: a block is linked to a video ONLY via its explicit
    name. Blocks whose first row has no video name are reported and NOT assigned to
    any video (we never infer names from row order).
    """
    raw = pd.read_csv(path, header=None, skiprows=2, dtype=str, keep_default_na=False)
    raw = raw.iloc[:, :13]
    raw.columns = ["video_name", "judge", "p1", "p2", "p3", "p4", "c1", "c2", "c3",
                   "total_mark", "score_100", "avg_score", "avg_score_rounded"]

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    blocks = []
    cur = None
    for i, r in raw.iterrows():
        if (r["judge"] or "").strip() == "" and all((r[c] or "").strip() == "" for c in raw.columns):
            continue
        starts_block = (r["avg_score"] or "").strip() != "" or (r["video_name"] or "").strip() != ""
        if starts_block or cur is None:
            cur = {"row": int(i) + 3, "video_name": r["video_name"].strip(), "rows": []}
            blocks.append(cur)
        cur["rows"].append(r)

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(empty slice|All-NaN|Degrees of freedom).*")
    records, issues = [], {"unnamed_blocks": [], "irregular_judge_count": [],
                           "average_mismatch": [], "duplicate_names": []}
    for b in blocks:
        rows = b["rows"]
        judges = [r["judge"].strip() for r in rows]
        items = np.array([[num(r[c]) for c in ["p1", "p2", "p3", "p4", "c1", "c2", "c3"]] for r in rows])
        s100 = np.array([num(r["score_100"]) for r in rows])
        avg = num(rows[0]["avg_score"])
        rec = {"video_name": b["video_name"], "csv_row": b["row"], "n_judges": len(rows),
               "judges": "|".join(judges), "avg_score": avg,
               "recomputed_avg": float(np.nanmean(s100)) if len(s100) else np.nan,
               "judge_sd": float(np.nanstd(s100)) if len(s100) else np.nan,
               "judge_range": float(np.nanmax(s100) - np.nanmin(s100)) if len(s100) else np.nan}
        for j in range(min(len(rows), judges_per_video)):
            rec[f"judge{j + 1}_score"] = s100[j]
        for k, c in enumerate(["p1", "p2", "p3", "p4", "c1", "c2", "c3"]):
            rec[f"mean_{c}"] = float(np.nanmean(items[:, k])) if len(items) else np.nan
        if not b["video_name"]:
            issues["unnamed_blocks"].append({"csv_row": b["row"], "avg_score": avg})
        if len(rows) != judges_per_video:
            issues["irregular_judge_count"].append({"video_name": b["video_name"], "csv_row": b["row"],
                                                    "n_judges": len(rows)})
        if np.isfinite(avg) and abs(avg - rec["recomputed_avg"]) > 0.01:
            issues["average_mismatch"].append({"video_name": b["video_name"], "csv_row": b["row"],
                                               "avg": avg, "recomputed": rec["recomputed_avg"]})
        records.append(rec)
    df = pd.DataFrame(records)
    named = df[df.video_name != ""]
    dup = named[named.duplicated("video_name", keep=False)]
    issues["duplicate_names"] = sorted(dup.video_name.unique().tolist())
    summary = {"n_blocks": len(df), "n_named_blocks": int(len(named)),
               "n_unnamed_blocks": len(issues["unnamed_blocks"]),
               "n_unique_named_videos": int(named.video_name.nunique()),
               "n_duplicate_names": len(issues["duplicate_names"]),
               "n_irregular_judge_count": len(issues["irregular_judge_count"]),
               "n_average_mismatch": len(issues["average_mismatch"]),
               "n_empty_residue_blocks": int((df.judges.str.strip("|") == "").sum()),
               "score_min": float(np.nanmin(named.avg_score)), "score_max": float(np.nanmax(named.avg_score)),
               "n_named_score_zero": int((named.avg_score == 0).sum()),
               "issues": issues}
    return df, summary
