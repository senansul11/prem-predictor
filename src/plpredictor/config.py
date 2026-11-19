"""Central configuration: paths, league code, seasons, URLs, seeds, simulation size."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "football_data_uk"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
REPORTS_DIR = ROOT / "reports"

LEAGUE = "E0"  # Premier League
SECOND_TIER = "E1"  # Championship (used for promoted-team priors)

FIRST_SEASON_START = 2015  # 2015-16
USER_AGENT = "plpredictor/0.1 (educational football modelling project)"
CSV_URL = "https://football-data.co.uk/mmz4281/{code}/{div}.csv"

SEED = 42
N_SIMS = 10_000
MATRIX_MAX_GOALS = 10  # score matrix covers 0..10 each side


def season_start_year(date: dt.date) -> int:
    """Year in which the season containing ``date`` started (season rolls over on 1 July)."""
    return date.year if date.month >= 7 else date.year - 1


def season_label(start_year: int) -> str:
    """2015 -> '2015-16'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_code(start_year: int) -> str:
    """2015 -> '1516' (football-data.co.uk folder name)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: dt.date | None = None) -> int:
    return season_start_year(today or dt.date.today())


def season_starts(today: dt.date | None = None) -> list[int]:
    """All season start years from FIRST_SEASON_START up to the current season."""
    return list(range(FIRST_SEASON_START, current_season_start(today) + 1))
