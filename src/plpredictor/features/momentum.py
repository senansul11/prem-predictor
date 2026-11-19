"""Rolling 5-match "momentum" per team, computed only from matches strictly before kickoff.

For each team we look at its last 5 matches (all divisions in the dataset, home and away,
so promoted teams bring their Championship form and early-season games use last season's
final games). Four components are each weighted 5,4,3,2,1 (most recent counts most):

* points per game
* goal difference per game
* shots-on-target difference per game
* over/under-performance: actual score minus Elo-expected score (win=1, draw=.5)

Each component is z-scored with the *expanding* mean/std of all earlier team-matches (so the
scaling never peeks forward), then averaged into one ``momentum`` score.
``mom_diff = home momentum - away momentum``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plpredictor.models.elo import EloParams, expected_home, run_elo

WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # oldest -> most recent of the last 5
COMPONENTS = ["ppg", "gd", "sot", "perf"]


def _team_rows(matches: pd.DataFrame, elo_params: EloParams) -> pd.DataFrame:
    """One row per team per played match with raw component values."""
    played = matches[matches["status"] == "played"]
    elo, _ = run_elo(matches, elo_params)
    elo.index = matches.index
    e = elo.loc[played.index]
    exp_h = expected_home(e["elo_h"].to_numpy(), e["elo_a"].to_numpy(), elo_params.home_adv)
    hg, ag = played["home_goals"].to_numpy(), played["away_goals"].to_numpy()
    s_h = np.where(hg > ag, 1.0, np.where(hg == ag, 0.5, 0.0))
    sot_diff = (played["home_sot"] - played["away_sot"]).to_numpy()
    base = {"date": played["date"].to_numpy(), "idx": played.index.to_numpy()}
    home = pd.DataFrame(
        {**base, "team": played["home_team"].to_numpy(), "ppg": 3 * (hg > ag) + (hg == ag),
         "gd": hg - ag, "sot": sot_diff, "perf": s_h - exp_h, "is_home": True}
    )  # fmt: skip
    away = pd.DataFrame(
        {**base, "team": played["away_team"].to_numpy(), "ppg": 3 * (ag > hg) + (hg == ag),
         "gd": ag - hg, "sot": -sot_diff, "perf": (1 - s_h) - (1 - exp_h), "is_home": False}
    )  # fmt: skip
    return pd.concat([home, away]).sort_values(["date", "idx"]).reset_index(drop=True)


def _weighted_prior_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    """For each position, weighted mean of the previous ``window`` values (NaN if none)."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        prev = values[max(0, i - window) : i]
        if len(prev) == 0:
            continue
        w = WEIGHTS[-len(prev) :]
        ok = ~np.isnan(prev)
        if ok.any():
            out[i] = np.sum(prev[ok] * w[ok]) / np.sum(w[ok])
    return out


def compute_momentum(matches: pd.DataFrame, elo_params: EloParams | None = None) -> pd.DataFrame:
    """Return ``mom_h``, ``mom_a``, ``mom_diff`` aligned to ``matches.index``."""
    elo_params = elo_params or EloParams()
    rows = _team_rows(matches, elo_params)
    for c in COMPONENTS:
        rows[f"r_{c}"] = rows.groupby("team")[c].transform(
            lambda s: pd.Series(_weighted_prior_mean(s.to_numpy()), index=s.index)
        )
    # z-score with expanding stats of earlier rows (shifted so a row never sees itself)
    z = []
    for c in COMPONENTS:
        r = rows[f"r_{c}"]
        mean = r.expanding(min_periods=50).mean().shift(1)
        std = r.expanding(min_periods=50).std().shift(1)
        z.append(((r - mean) / std).clip(-4, 4))
    rows["momentum"] = pd.concat(z, axis=1).mean(axis=1, skipna=True).fillna(0.0)

    out = pd.DataFrame(index=matches.index, columns=["mom_h", "mom_a"], dtype=float)
    h = rows[rows["is_home"]].set_index("idx")["momentum"]
    a = rows[~rows["is_home"]].set_index("idx")["momentum"]
    out.loc[h.index, "mom_h"] = h
    out.loc[a.index, "mom_a"] = a
    out = out.fillna(0.0)
    out["mom_diff"] = out["mom_h"] - out["mom_a"]
    return out
