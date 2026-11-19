"""Elo ratings across the Premier League and Championship on one common scale.

Teams keep their rating when they move between divisions, so a promoted club arrives with a
rating earned in the Championship. Each division regresses part-way to its own mean at the
start of every season; the Championship mean sits ``league_gap`` points below the PL mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = 1500.0


@dataclass
class EloParams:
    k: float = 20.0
    home_adv: float = 70.0
    regress: float = 0.3  # fraction pulled to the division mean each new season
    league_gap: float = 150.0  # PL mean minus Championship mean
    gd_scale: float = 1.0  # 0 disables the goal-difference multiplier


def expected_home(r_home: float, r_away: float, home_adv: float) -> float:
    """Expected score (win=1, draw=0.5) of the home side."""
    return 1.0 / (1.0 + 10 ** (-(r_home + home_adv - r_away) / 400.0))


def update(
    r_home: float, r_away: float, hg: int, ag: int, p: EloParams
) -> tuple[float, float]:
    """Return new (home, away) ratings after one match. The change is zero-sum."""
    exp = expected_home(r_home, r_away, p.home_adv)
    score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    gd = abs(hg - ag)
    mult = 1.0 + p.gd_scale * np.log1p(max(gd - 1, 0)) if gd > 1 else 1.0
    delta = p.k * mult * (score - exp)
    return r_home + delta, r_away - delta


def division_mean(div: str, p: EloParams) -> float:
    return BASE if div == "E0" else BASE - p.league_gap


def run_elo(matches: pd.DataFrame, p: EloParams | None = None):
    """Walk through played matches in date order.

    Returns (frame, final_ratings). ``frame`` has the pre-match ratings for every played
    match (``elo_h``, ``elo_a``) aligned to ``matches.index`` so no future data is used.
    """
    p = p or EloParams()
    played = matches[matches["status"] == "played"].sort_values(["date", "home_team"])
    ratings: dict[str, float] = {}
    last_season: str | None = None
    elo_h = np.full(len(matches), np.nan)
    elo_a = np.full(len(matches), np.nan)
    pos = {idx: i for i, idx in enumerate(matches.index)}

    for season, sdf in played.groupby("season", sort=True):
        # new season: regress every team in this season towards its division's mean
        team_div = {}
        for div, g in sdf.groupby("div"):
            for t in set(g["home_team"]) | set(g["away_team"]):
                team_div[t] = div
        for t, div in team_div.items():
            mean = division_mean(div, p)
            if t not in ratings:
                ratings[t] = mean
            elif last_season is not None:
                ratings[t] = mean + (1 - p.regress) * (ratings[t] - mean)
        last_season = season
        for idx, row in sdf.iterrows():
            h, a = row["home_team"], row["away_team"]
            rh, ra = ratings[h], ratings[a]
            elo_h[pos[idx]], elo_a[pos[idx]] = rh, ra
            ratings[h], ratings[a] = update(rh, ra, row["home_goals"], row["away_goals"], p)

    out = matches[["match_id"]].copy()
    out["elo_h"], out["elo_a"] = elo_h, elo_a
    return out, ratings


class EloProbModel:
    """Turn an Elo difference into H/D/A probabilities with a multinomial logistic fit.

    Elo itself only gives an expected score; a small logistic regression on the rating
    difference learns how often draws happen for close vs lopsided matches.
    """

    def __init__(self, home_adv: float = 70.0) -> None:
        self.home_adv = home_adv
        self.lr = LogisticRegression(C=100.0, max_iter=500)

    def _x(self, elo_h, elo_a) -> np.ndarray:
        d = (np.asarray(elo_h) + self.home_adv - np.asarray(elo_a)) / 100.0
        return np.column_stack([d, np.abs(d)])

    def fit(self, elo_h, elo_a, y) -> EloProbModel:
        self.lr.fit(self._x(elo_h, elo_a), y)
        return self

    def predict(self, elo_h, elo_a) -> np.ndarray:
        raw = self.lr.predict_proba(self._x(elo_h, elo_a))
        out = np.zeros((len(raw), 3))
        out[:, self.lr.classes_] = raw
        return out
