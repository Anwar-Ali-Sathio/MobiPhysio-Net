"""Algorithm 1 - VIDEO_PREPROCESSING as a stream-process-delete pipeline.

For every inventory video: download (resumable, checksum-verified) -> decode T
uniformly spaced RGB frames -> letterbox to S x S -> cache uint8 frames + mask ->
optionally delete the raw MP4. Each video writes a status record so the job is
resumable and every exclusion has a recorded reason.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..dataset.download import download_file
from .cache import FrameCache
from .frames import read_sampled_frames

log = logging.getLogger(__name__)


def process_one(rec: dict, data_dir: str, t: int, size: int, dl_cfg: dict) -> dict:
    import cv2
    cv2.setNumThreads(1)
    cache = FrameCache(Path(data_dir), t, size)
    vid = Path(rec["filename"]).stem
    prev = cache.read_status(vid)
    if prev and prev.get("status") == "ok" and cache.has(vid):
        prev["skipped_existing"] = True
        return prev
    raw = Path(data_dir) / "raw" / rec["filename"]
    t0 = time.time()
    st = {"video_id": vid, "file_id": rec["file_id"], "filename": rec["filename"],
          "expected_size": rec["size_bytes"], "expected_md5": rec["checksum"]}
    d = download_file(rec["download_url"], raw, rec["size_bytes"], rec["checksum"],
                      rec["checksum_type"], timeout_s=dl_cfg["timeout_s"],
                      max_retries=dl_cfg["max_retries"], backoff_s=dl_cfg["backoff_s"],
                      chunk_bytes=dl_cfg["chunk_bytes"])
    st.update(download_status=d["status"], download_attempts=d["attempts"],
              download_error=d["error"], download_resumed=d["resumed"],
              checksum_verified=d["status"] == "ok", download_s=round(time.time() - t0, 2))
    if d["status"] != "ok":
        st.update(status="download_failed", decode_status="not_attempted")
        cache.write_status(vid, st)
        return st
    t1 = time.time()
    try:
        r = read_sampled_frames(str(raw), t, size)
    except Exception as exc:  # decoder crash -> recorded exclusion
        r = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    st["decode_s"] = round(time.time() - t1, 2)
    if not r["ok"]:
        st.update(status="decode_failed", decode_status="failed", decode_error=r["error"])
    else:
        cache.save(vid, r["frames"], r["mask"])
        st.update(status="ok", decode_status="ok",
                  **{k: r.get(k) for k in ["indices", "timestamps", "reported_frames",
                                           "decoded_frames", "fps", "width", "height",
                                           "rotation", "decoded_shape", "n_valid", "padded",
                                           "decoder", "pyav_error"]})
    if not dl_cfg.get("keep_raw_videos", False) and st["status"] == "ok":
        raw.unlink(missing_ok=True)
    st["raw_retained"] = raw.exists()
    cache.write_status(vid, st)
    return st


def run_preprocessing(inventory: list[dict], data_dir: Path, t: int, size: int,
                      dl_cfg: dict, workers: int = 4, limit: int | None = None,
                      video_ids: set[str] | None = None, shuffle_seed: int = 0) -> list[dict]:
    vids = [r for r in inventory if (r["content_type"] or "").startswith("video/")]
    if video_ids is not None:
        vids = [r for r in vids if Path(r["filename"]).stem in video_ids]
    # Seeded shuffle: any prefix of the job is a class/participant-mixed sample, so a
    # pilot can run before the full (multi-hour) pass finishes.
    order = np.random.default_rng(shuffle_seed).permutation(len(vids))
    vids = [vids[i] for i in order]
    if limit:
        vids = vids[:limit]
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, r, str(data_dir), t, size, dl_cfg): r for r in vids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
            except Exception as exc:
                r = futs[f]
                res = {"video_id": Path(r["filename"]).stem, "status": "worker_error",
                       "error": repr(exc)}
            results.append(res)
            if i % 25 == 0 or i == len(vids):
                ok = sum(x.get("status") == "ok" for x in results)
                log.info("preprocess %d/%d ok=%d elapsed=%.0fs", i, len(vids), ok, time.time() - t0)
    return results
