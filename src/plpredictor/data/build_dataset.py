"""Build ``matches.parquet`` from cached CSVs (idempotent upsert on ``match_id``)."""

from __future__ import annotations

import pandas as pd

from plpredictor import config
from plpredictor.data import schema
from plpredictor.data.sources import football_data_uk as fduk
from plpredictor.data.validate import check_not_truncated, validate_matches

MATCHES_PATH = config.PROCESSED_DIR / "matches.parquet"
COVERAGE_PATH = config.PROCESSED_DIR / "data_coverage.csv"


def upsert(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Insert/replace rows by ``match_id``. Re-running with identical data changes nothing."""
    if existing is None or existing.empty:
        merged = new.copy()
    else:
        merged = pd.concat([existing[~existing["match_id"].isin(new["match_id"])], new])
    return merged.sort_values(["div", "date", "home_team"]).reset_index(drop=True)


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per division/season: rows and % coverage of xG, Pinnacle odds, closing odds, etc."""
    g = df.groupby(["div", "season"])
    rep = pd.DataFrame(
        {
            "rows": g.size(),
            "played": g["status"].apply(lambda s: (s == "played").sum()),
            "pct_xg": g["home_xg"].apply(lambda s: 100 * s.notna().mean()),
            "pct_pre_odds": g["p_mkt_h"].apply(lambda s: 100 * s.notna().mean()),
            "pct_pinnacle_pre": g["mkt_src"].apply(lambda s: 100 * (s == "pinnacle").mean()),
            "pct_closing": g["p_close_h"].apply(lambda s: 100 * s.notna().mean()),
            "pct_pinnacle_close": g["close_src"].apply(lambda s: 100 * (s == "pinnacle").mean()),
            "pct_over25": g["p_over25"].apply(lambda s: 100 * s.notna().mean()),
            "pct_ah": g["ah_line"].apply(lambda s: 100 * s.notna().mean()),
        }
    )
    return rep.round(1).reset_index()


def build(divs: tuple[str, ...] | None = None, today=None) -> pd.DataFrame:
    """Read every cached season for each division, clean, validate and write parquet."""
    divs = divs or (config.LEAGUE, config.SECOND_TIER)
    frames = []
    for div in divs:
        for start in config.season_starts(today):
            raw = fduk.load_cached(start, div)
            if raw is None:
                continue
            frames.append(schema.clean_season(schema.read_raw_csv(raw), start, div))
    if not frames:
        raise FileNotFoundError("No cached CSVs found; run with --download first.")
    new = pd.concat(frames, ignore_index=True)

    for div, g in new.groupby("div"):
        # 2026-27 is a partial season, so only check team counts on completed seasons.
        full = g[g["season"] != config.season_label(config.current_season_start(today))]
        teams = 20 if div == config.LEAGUE else 24
        problems = validate_matches(g, expect_teams=None) + validate_matches(
            full, expect_teams=teams
        )
        if problems:
            raise ValueError(f"Validation failed for {div}: {sorted(set(problems))}")

    existing = pd.read_parquet(MATCHES_PATH) if MATCHES_PATH.exists() else None
    problems = check_not_truncated(existing, new)
    if problems:
        raise ValueError(f"Refusing to publish: {problems}")
    merged = upsert(existing, new)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MATCHES_PATH, index=False)
    coverage_report(merged).to_csv(COVERAGE_PATH, index=False)
    return merged


def load_matches(div: str | None = None) -> pd.DataFrame:
    """Load processed matches (optionally one division), sorted by date."""
    from plpredictor.features import xg_proxy

    df = pd.read_parquet(MATCHES_PATH)
    df = xg_proxy.add_proxy(df, xg_proxy.load_or_fit(df))  # shot-based xG proxy columns
    if div:
        df = df[df["div"] == div]
    return df.sort_values(["date", "home_team"]).reset_index(drop=True)
