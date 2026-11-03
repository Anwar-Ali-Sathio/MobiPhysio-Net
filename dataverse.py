"""Minimal Harvard Dataverse API client (metadata + file inventory)."""
from __future__ import annotations

import time
from typing import Any

import requests

# Harvard Dataverse returns HTTP 403 for the default ``python-requests`` User-Agent
# (verified 2026-10-27), so an explicit, identifying UA is always sent.
USER_AGENT = "mobiphysio-research/0.1 (+https://doi.org/10.7910/DVN/XSI0QN)"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


class DataverseError(RuntimeError):
    pass


class DataverseClient:
    def __init__(self, api_base: str = "https://dataverse.harvard.edu/api",
                 timeout_s: float = 60, max_retries: int = 5, backoff_s: float = 2.0,
                 session: requests.Session | None = None):
        self.api_base = api_base.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.session = session or make_session()

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_s)
                if r.status_code >= 500 or r.status_code == 429:
                    raise DataverseError(f"HTTP {r.status_code} for {url}")
                if r.status_code != 200:
                    # 4xx is not retryable.
                    raise DataverseError(f"HTTP {r.status_code} for {url}: {r.text[:300]}")
                payload = r.json()
                if payload.get("status") != "OK":
                    raise DataverseError(f"Dataverse status {payload.get('status')}: {payload}")
                return payload
            except (requests.RequestException, DataverseError) as exc:
                last = exc
                if isinstance(exc, DataverseError) and "HTTP 4" in str(exc) and "429" not in str(exc):
                    raise
                time.sleep(self.backoff_s * (2 ** attempt))
        raise DataverseError(f"Failed after {self.max_retries} attempts: {last}")

    def resolve_doi(self, doi: str) -> str:
        """Return the landing page that https://doi.org/<doi> redirects to."""
        bare = doi.replace("doi:", "")
        r = self.session.head(f"https://doi.org/{bare}", allow_redirects=False, timeout=self.timeout_s)
        return r.headers.get("Location", "")

    def dataset(self, doi: str) -> dict[str, Any]:
        return self._get_json(f"{self.api_base}/datasets/:persistentId/",
                              params={"persistentId": doi})["data"]

    def access_url(self, file_id: int) -> str:
        return f"{self.api_base}/access/datafile/{file_id}"


def parse_inventory(dataset: dict[str, Any], api_base: str) -> list[dict[str, Any]]:
    """Flatten the latest-version file list into inventory records."""
    version = dataset["latestVersion"]
    rows = []
    for f in version.get("files", []):
        df = f.get("dataFile", {})
        checksum = df.get("checksum") or {}
        rows.append({
            "file_id": df.get("id"),
            "filename": df.get("filename") or f.get("label"),
            "label": f.get("label"),
            "directory": f.get("directoryLabel") or "",
            "content_type": df.get("contentType"),
            "size_bytes": df.get("filesize"),
            "checksum_type": checksum.get("type") or ("MD5" if df.get("md5") else None),
            "checksum": checksum.get("value") or df.get("md5"),
            "restricted": bool(f.get("restricted", False)),
            "persistent_id": df.get("persistentId") or "",
            "creation_date": df.get("creationDate"),
            "download_url": f"{api_base.rstrip('/')}/access/datafile/{df.get('id')}",
        })
    return rows


def summarize_metadata(dataset: dict[str, Any], doi: str, landing_page: str) -> dict[str, Any]:
    v = dataset["latestVersion"]
    fields = {f["typeName"]: f["value"] for f in v["metadataBlocks"]["citation"]["fields"]}

    def _val(x):
        return x.get("value") if isinstance(x, dict) else x

    authors = [_val(a.get("authorName")) for a in fields.get("author", [])]
    desc = [_val(d.get("dsDescriptionValue")) for d in fields.get("dsDescription", [])]
    return {
        "doi": doi,
        "landing_page": landing_page,
        "dataset_id": dataset.get("id"),
        "persistent_url": dataset.get("persistentUrl"),
        "publisher": dataset.get("publisher"),
        "title": fields.get("title"),
        "authors": authors,
        "description": desc,
        "version": f"{v.get('versionNumber')}.{v.get('versionMinorNumber')}",
        "version_state": v.get("versionState"),
        "release_time": v.get("releaseTime"),
        "last_update_time": v.get("lastUpdateTime"),
        "license": v.get("license"),
        "terms_of_use": v.get("termsOfUse"),
        "n_files": len(v.get("files", [])),
        "deposit_date": fields.get("dateOfDeposit"),
    }
