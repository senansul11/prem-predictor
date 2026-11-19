"""football-data.co.uk season CSVs: download with cache, or read straight from the cache."""

from __future__ import annotations

import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from plpredictor import config


def cache_path(start_year: int, div: str) -> Path:
    return config.RAW_DIR / div / f"{config.season_code(start_year)}.csv"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def download_season(start_year: int, div: str, force: bool = False, pause: float = 1.0) -> Path:
    """Download one season CSV into the cache (skipped if cached and not ``force``)."""
    path = cache_path(start_year, div)
    if path.exists() and not force:
        return path
    url = config.CSV_URL.format(code=config.season_code(start_year), div=div)
    resp = _session().get(url, timeout=30)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    time.sleep(pause)  # be polite
    return path


def load_cached(start_year: int, div: str) -> bytes | None:
    """Raw bytes from the cache, or None if this season is not cached (offline mode)."""
    path = cache_path(start_year, div)
    return path.read_bytes() if path.exists() else None


def fetch_bytes(start_year: int, div: str) -> bytes:
    """Download one season CSV and return it WITHOUT touching the cache (for change detection)."""
    url = config.CSV_URL.format(code=config.season_code(start_year), div=div)
    resp = _session().get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def write_cache(
    start_year: int, div: str, content: bytes, snapshot_date: str | None = None
) -> None:
    """Save a downloaded CSV to the cache, plus a dated snapshot so every run is reproducible."""
    path = cache_path(start_year, div)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if snapshot_date:
        snap = path.parent / "snapshots" / snapshot_date / path.name
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(content)
