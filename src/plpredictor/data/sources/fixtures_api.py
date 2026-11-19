"""footballdata.io fixtures client (full remaining schedule with dates).

Verified against the live API: base URL ``https://footballdata.io/api/v1``, Bearer auth,
``/fixtures/upcoming`` with ``league_id`` (Premier League = 15), ``limit`` (max 100) and
``page``. The free plan allows 1,000 requests a month, so every response is cached on disk.
The response shape varies: ``data`` is a list on page 1 but ``{"matches": [...]}`` on page 2,
so ``extract_matches`` accepts both. The key is read from ``.env`` and never logged.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from plpredictor import config
from plpredictor.data.team_names import canonical

BASE_URL = "https://footballdata.io/api/v1"
PL_LEAGUE_ID = 15
CACHE_DIR = config.DATA_DIR / "raw" / "fixtures_api"
ENV_PATH = config.ROOT / ".env"


class MissingKeyError(RuntimeError):
    """Raised when FOOTBALLDATA_IO_API_KEY is not set."""


def get_api_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.getenv("FOOTBALLDATA_IO_API_KEY")
    if not key:
        raise MissingKeyError("Add FOOTBALLDATA_IO_API_KEY to the .env file (see .env.example).")
    return key


def extract_matches(payload: dict) -> list[dict]:
    """Return the match list from either response shape."""
    data = payload["data"]
    return data if isinstance(data, list) else data["matches"]


def _odds(m: dict, key: str) -> float:
    """API odds of 0 mean 'not available': treat as missing, never as zero."""
    v = (m.get("odds") or {}).get(key)
    return float(v) if v and v > 1 else float("nan")


def parse_fixtures(matches: list[dict]) -> pd.DataFrame:
    """Matches -> frame with canonical team names (raises on an unmapped team)."""
    rows = [
        {
            "date": pd.Timestamp(m["match_date"]),
            "game_week": m.get("game_week"),
            "home_team": canonical(m["home_team"]["team_name"]),
            "away_team": canonical(m["away_team"]["team_name"]),
            "status": m.get("status_localized") or m.get("status"),
            "api_match_id": m["match_id"],
            "api_odds_h": _odds(m, "home_win"),
            "api_odds_d": _odds(m, "draw"),
            "api_odds_a": _odds(m, "away_win"),
        }
        for m in matches
    ]
    return pd.DataFrame(rows).sort_values(["date", "home_team"]).reset_index(drop=True)


def fetch_upcoming(to: str | None = None, pause: float = 1.0) -> list[dict]:
    """Download all upcoming Premier League fixtures (paged) and save a dated snapshot."""
    import time

    headers = {"Authorization": f"Bearer {get_api_key()}", "User-Agent": config.USER_AGENT}
    to = to or f"{config.current_season_start() + 1}-07-31"
    out: list[dict] = []
    page = 1
    while True:
        params = {"league_id": PL_LEAGUE_ID, "limit": 100, "page": page, "to": to}
        resp = requests.get(f"{BASE_URL}/fixtures/upcoming", headers=headers, params=params,
                            timeout=30)  # fmt: skip
        resp.raise_for_status()
        payload = resp.json()
        out += extract_matches(payload)
        pages = payload.get("meta", {}).get("pagination", {}).get("total_pages", 1)
        if page >= pages:
            break
        page += 1
        time.sleep(pause)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    (CACHE_DIR / f"upcoming_{stamp}.json").write_text(json.dumps(out), encoding="utf-8")
    (CACHE_DIR / "upcoming_latest.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def load_cached() -> pd.DataFrame | None:
    """Latest cached fixtures (works offline), or None if never fetched."""
    path = CACHE_DIR / "upcoming_latest.json"
    if not path.exists():
        return None
    return parse_fixtures(json.loads(path.read_text(encoding="utf-8")))


def full_remaining(played: pd.DataFrame, teams: list[str], dated: pd.DataFrame | None):
    """All remaining fixtures: dated ones from the API plus any undated pairs still owed.

    Pairs the API does not list yet (the schedule is published in stages) are kept with
    ``date = NaT`` so the simulation still covers the whole season.
    """
    from plpredictor.simulation.season_sim import remaining_fixtures

    owed = remaining_fixtures(played, teams)
    if dated is None or dated.empty:
        return owed.assign(date=pd.NaT, game_week=np_nan(), dated=False)
    dated = dated[dated["home_team"].isin(teams) & dated["away_team"].isin(teams)]
    dated = dated.drop_duplicates(["home_team", "away_team"]).assign(dated=True)
    missing = owed.merge(dated[["home_team", "away_team"]], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    undated = missing.assign(date=pd.NaT, game_week=np_nan(), dated=False)
    keep = owed.merge(dated, on=["home_team", "away_team"], how="inner")
    return pd.concat([keep, undated], ignore_index=True)


def np_nan() -> float:
    return float("nan")
