"""Validation gates (spec 8.3). Each returns a list of problems; empty means OK."""

from __future__ import annotations

import pandas as pd


def validate_matches(df: pd.DataFrame, expect_teams: int | None = 20) -> list[str]:
    """Check one division's cleaned matches. ``expect_teams`` applies per season."""
    problems: list[str] = []
    if df["match_id"].duplicated().any():
        problems.append(f"{df['match_id'].duplicated().sum()} duplicate match_id values")

    played = df[df["status"] == "played"]
    for col in ("home_goals", "away_goals"):
        v = played[col]
        if (v < 0).any() or (v % 1 != 0).any() or v.isna().any():
            problems.append(f"{col} has negative, fractional or missing values")
    exp = pd.Series("D", index=played.index)
    exp[played["home_goals"] > played["away_goals"]] = "H"
    exp[played["home_goals"] < played["away_goals"]] = "A"
    if (exp != played["result"]).any():
        problems.append("result does not match the score")

    long = pd.concat(
        [
            df[["season", "date", "home_team"]].rename(columns={"home_team": "team"}),
            df[["season", "date", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    )
    if long.duplicated().any():
        problems.append("a team plays twice on the same date")

    if expect_teams is not None:
        for season, g in df.groupby("season"):
            n = len(set(g["home_team"]) | set(g["away_team"]))
            if n != expect_teams:
                problems.append(f"{season}: {n} teams, expected {expect_teams}")
    return problems


def check_not_truncated(existing: pd.DataFrame | None, new: pd.DataFrame) -> list[str]:
    """Gate: the number of played matches per division/season must never go down.

    A truncated or partly broken download would have fewer played matches than what we already
    hold, so the refresh stops instead of publishing it.
    """
    if existing is None or existing.empty:
        return []

    def played(df: pd.DataFrame) -> pd.Series:
        return df[df["status"] == "played"].groupby(["div", "season"]).size()

    old, cur = played(existing), played(new)
    problems = []
    for key, n_old in old.items():
        if key in cur.index and cur[key] < n_old:
            problems.append(
                f"{key[0]} {key[1]}: {cur[key]} played matches, previously {n_old} "
                "(truncated download?)"
            )
    return problems
