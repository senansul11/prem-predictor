"""Leak-free match features. Every value uses only matches strictly before kickoff.

Feature groups: Elo, momentum, rolling form (10 matches, plus venue-specific 5), rest days
and walk-forward Dixon-Coles ratings. Rolling statistics are computed per team over the team's
*previous* rows (``shift(1)``), so a match never sees itself or anything after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.features.momentum import compute_momentum
from plpredictor.models.dixon_coles import DCConfig, DixonColes
from plpredictor.models.elo import EloParams, run_elo

DC_FEATS = ["dc_lam_h", "dc_lam_a", "dc_att_h", "dc_def_h", "dc_att_a", "dc_def_a",
            "dc_p_h", "dc_p_d", "dc_p_a"]  # fmt: skip
ELO_FEATS = ["elo_h", "elo_a", "elo_diff", "elo_chg5_h", "elo_chg5_a"]
MOM_FEATS = ["mom_h", "mom_a", "mom_diff"]
ODDS_FEATS = ["p_mkt_h", "p_mkt_d", "p_mkt_a"]
FORM_STATS = ["ppg", "gf", "ga", "sh", "sot", "sot_ag"]
VENUE_STATS = ["ppg", "gf", "ga"]


def form_features() -> list[str]:
    cols = []
    for side in ("h", "a"):
        cols += [f"f10_{s}_{side}" for s in FORM_STATS]
        cols += [f"v5_{s}_{side}" for s in VENUE_STATS]
        cols += [f"rest_{side}"]
    return cols


def team_rows(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played match, with that match's raw values and pre-match Elo."""
    p = matches[matches["status"] == "played"]
    e = elo.loc[p.index]
    hg, ag = p["home_goals"].to_numpy(), p["away_goals"].to_numpy()

    def side(home: bool) -> pd.DataFrame:
        f, a = (hg, ag) if home else (ag, hg)
        sh, sh_a = ("home_shots", "away_shots") if home else ("away_shots", "home_shots")
        st, st_a = ("home_sot", "away_sot") if home else ("away_sot", "home_sot")
        return pd.DataFrame(
            {
                "idx": p.index.to_numpy(),
                "date": p["date"].to_numpy(),
                "team": p["home_team" if home else "away_team"].to_numpy(),
                "is_home": home,
                "ppg": 3 * (f > a) + (f == a),
                "gf": f, "ga": a,
                "sh": p[sh].to_numpy(), "sot": p[st].to_numpy(), "sot_ag": p[st_a].to_numpy(),
                "elo_pre": e["elo_h" if home else "elo_a"].to_numpy(),
            }
        )  # fmt: skip

    return pd.concat([side(True), side(False)]).sort_values(["date", "idx"]).reset_index(drop=True)


def _rolling_prior(g: pd.core.groupby.SeriesGroupBy, window: int, min_periods: int):
    return g.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())


def team_form(rows: pd.DataFrame) -> pd.DataFrame:
    """Per team-row prior-only rolling stats, rest days and Elo change over 5 matches."""
    r = rows.copy()
    by_team = r.groupby("team")
    for s in FORM_STATS:
        r[f"f10_{s}"] = _rolling_prior(by_team[s], 10, 3)
    by_venue = r.groupby(["team", "is_home"])
    for s in VENUE_STATS:
        r[f"v5_{s}"] = _rolling_prior(by_venue[s], 5, 2)
    r["rest"] = by_team["date"].diff().dt.days.clip(upper=14)
    r["elo_chg5"] = r["elo_pre"] - by_team["elo_pre"].shift(5)
    return r


def build_features(
    matches: pd.DataFrame,
    elo_params: EloParams,
    dc_cfg: DCConfig,
    dc_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature frame aligned to ``matches.index`` (rows for E0 and E1; use E0 for modelling)."""
    elo, _ = run_elo(matches, elo_params)
    elo.index = matches.index
    rows = team_form(team_rows(matches, elo))
    stat_cols = (
        [f"f10_{s}" for s in FORM_STATS] + [f"v5_{s}" for s in VENUE_STATS] + ["rest", "elo_chg5"]
    )
    out = pd.DataFrame(index=matches.index)
    for home, suffix in ((True, "h"), (False, "a")):
        sub = rows[rows["is_home"] == home].set_index("idx")[stat_cols]
        sub.columns = [f"{c}_{suffix}" for c in sub.columns]
        out = out.join(sub)
    out["elo_h"], out["elo_a"] = elo["elo_h"], elo["elo_a"]
    out["elo_diff"] = out["elo_h"] - out["elo_a"]
    out = out.rename(columns={"elo_chg5_h": "elo_chg5_h", "elo_chg5_a": "elo_chg5_a"})
    out = out.join(compute_momentum(matches, elo_params))
    if dc_features is None:
        dc_features = walk_forward_dc_features(matches, dc_cfg)
    return out.join(dc_features)


def walk_forward_dc_features(
    matches: pd.DataFrame, cfg: DCConfig, first_season: str = "2016-17", refit_days: int = 7
) -> pd.DataFrame:
    """Dixon-Coles ratings/probabilities for each E0 match from a model fitted on prior data."""
    from plpredictor.evaluation.backtest import chunk_by_days

    played = matches[matches["status"] == "played"]
    model = DixonColes(cfg)
    parts = []
    for season in sorted(played["season"].unique()):
        if season < first_season:
            continue
        test = played[(played["season"] == season) & (played["div"] == config.LEAGUE)]
        for chunk in chunk_by_days(test, refit_days):
            train = played[played["date"] < chunk["date"].min()]
            model.fit(train, ref_date=chunk["date"].min())
            f = model.fit_
            idx = f.index
            pr = model.predict(chunk)
            lam = pr["xg_h"].to_numpy(), pr["xg_a"].to_numpy()

            def get(arr, teams, idx=idx):
                return np.array([arr[idx[t]] if t in idx else 0.0 for t in teams])

            parts.append(
                pd.DataFrame(
                    {
                        "dc_lam_h": lam[0], "dc_lam_a": lam[1],
                        "dc_att_h": get(f.attack, chunk["home_team"]),
                        "dc_def_h": get(f.defence, chunk["home_team"]),
                        "dc_att_a": get(f.attack, chunk["away_team"]),
                        "dc_def_a": get(f.defence, chunk["away_team"]),
                        "dc_p_h": pr["p_h"].to_numpy(), "dc_p_d": pr["p_d"].to_numpy(),
                        "dc_p_a": pr["p_a"].to_numpy(),
                    },
                    index=chunk.index,
                )  # fmt: skip
            )
    return pd.concat(parts).reindex(matches.index)
