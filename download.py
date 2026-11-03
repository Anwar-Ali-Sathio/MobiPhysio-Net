"""Resumable, checksum-verified file download."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import requests

from .dataverse import make_session


def file_checksum(path: str | Path, algo: str = "md5", chunk: int = 1 << 22) -> str:
    h = hashlib.new(algo.lower())
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: str | Path, size: int | None, checksum: str | None,
                checksum_type: str | None = "MD5") -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, "missing"
    if size is not None and path.stat().st_size != size:
        return False, f"size mismatch: {path.stat().st_size} != {size}"
    if checksum:
        got = file_checksum(path, (checksum_type or "md5").replace("-", ""))
        if got.lower() != checksum.lower():
            return False, f"checksum mismatch: {got} != {checksum}"
    return True, "ok"


def download_file(url: str, dest: str | Path, size: int | None = None,
                  checksum: str | None = None, checksum_type: str | None = "MD5",
                  timeout_s: float = 120, max_retries: int = 5, backoff_s: float = 2.0,
                  chunk_bytes: int = 1 << 22, session: requests.Session | None = None) -> dict:
    """Download ``url`` to ``dest`` with HTTP Range resume, retries and verification.

    Returns a status record; never raises for network errors (they are reported).
    An existing ``dest`` that passes verification is not downloaded again.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    sess = session or make_session()
    status = {"url": url, "dest": str(dest), "status": "failed", "attempts": 0,
              "bytes": 0, "error": "", "skipped_existing": False, "resumed": False}

    if dest.exists():
        ok, why = verify_file(dest, size, checksum, checksum_type)
        if ok:
            status.update(status="ok", bytes=dest.stat().st_size, skipped_existing=True)
            return status
        dest.unlink()

    for attempt in range(1, max_retries + 1):
        status["attempts"] = attempt
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with sess.get(url, headers=headers, stream=True, timeout=timeout_s,
                          allow_redirects=True) as r:
                if r.status_code == 416 and size is not None and have == size:
                    pass  # already complete
                elif r.status_code not in (200, 206):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                else:
                    mode = "ab" if (have and r.status_code == 206) else "wb"
                    status["resumed"] = status["resumed"] or mode == "ab"
                    with open(part, mode) as fh:
                        for block in r.iter_content(chunk_size=chunk_bytes):
                            if block:
                                fh.write(block)
            ok, why = verify_file(part, size, checksum, checksum_type)
            if ok:
                os.replace(part, dest)
                status.update(status="ok", bytes=dest.stat().st_size, error="")
                return status
            status["error"] = why
            if "checksum" in why or (size is not None and part.stat().st_size > size):
                part.unlink(missing_ok=True)  # corrupt; restart from zero
        except (requests.RequestException, OSError) as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(backoff_s * (2 ** (attempt - 1)))
    return status
