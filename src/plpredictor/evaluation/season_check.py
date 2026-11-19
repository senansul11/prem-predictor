# ruff: noqa: E501
"""Season-level check: simulate from mid-season of past seasons and score the odds.

For each past season and cut-off (after ~10 and ~20 games per team), fit the model on earlier
data, simulate the rest of the season (with and without parameter uncertainty) and compare
the title / top-4 / relegation probabilities with what actually happened (Brier score).
Run: python -m plpredictor.evaluation.season_check
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.models.dixon_coles import (
    DCConfig,
    DixonColes,
    expected_goals,
    parameter_draws,
    score_matrices,
)
from plpredictor.simulation import season_sim

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
CUTOFF_MATCHES = (100, 200)  # played matches in the season at the cut-off (~10 / ~20 each)
SCALES = (0.0, 0.5, 1.0, 1.5)  # 0 = point estimate (no uncertainty)


def events(final_table: pd.DataFrame) -> pd.DataFrame:
    """Actual outcomes per team: title / top4 / relegated (0 or 1)."""
    pos = np.arange(len(final_table))
    return pd.DataFrame(
        {"title": (pos == 0), "top4": (pos < 4), "relegated": (pos >= len(pos) - 3)},
        index=final_table.index,
    ).astype(float)


def run(n_sims: int = 4000, k: int = 100) -> pd.DataFrame:
    matches = build_dataset.load_matches()
    played = matches[matches["status"] == "played"]
    cfg = DCConfig(**json.loads((config.OUTPUTS_DIR / "dc_config.json").read_text()))
    rows = []
    for season in SEASONS:
        s = played[(played["season"] == season) & (played["div"] == config.LEAGUE)]
        s = s.sort_values(["date", "home_team"])
        teams = sorted(set(s["home_team"]) | set(s["away_team"]))
        actual = events(season_sim.current_table(s, teams))
        for n_cut in CUTOFF_MATCHES:
            cut_date = s["date"].iloc[n_cut - 1]
            seen = s[s["date"] <= cut_date]
            train = played[played["date"] <= cut_date]
            ref = cut_date + pd.Timedelta(days=1)
            model = DixonColes(cfg).fit(train, ref_date=ref)
            fixtures = season_sim.remaining_fixtures(seen, teams)
            table = season_sim.current_table(seen, teams)
            for scale in SCALES:
                if scale == 0.0:
                    mats = model.matrices(fixtures)
                    res = season_sim.simulate(table, fixtures, mats, n_sims=n_sims)
                else:
                    draws = parameter_draws(train, ref, cfg, model.fit_, k=k, scale=scale)
                    per = []
                    for d in draws:
                        lam, mu = expected_goals(d, fixtures["home_team"], fixtures["away_team"])
                        per.append(score_matrices(lam, mu, d.rho, cfg.max_goals))
                    res = season_sim.simulate_uncertain(table, fixtures, per, n_sims=n_sims)
                p = res.summary().set_index("team").loc[teams]
                pred = pd.DataFrame(
                    {"title": p["p_title"], "top4": p["p_top4"], "relegated": p["p_relegation"]}
                )
                for ev in ("title", "top4", "relegated"):
                    rows.append(
                        {
                            "season": season, "cutoff": n_cut, "scale": scale, "event": ev,
                            "brier": float(np.mean((pred[ev] - actual[ev]) ** 2)),
                            "log_loss": float(
                                -np.mean(
                                    actual[ev] * np.log(np.clip(pred[ev], 1e-3, 1))
                                    + (1 - actual[ev]) * np.log(np.clip(1 - pred[ev], 1e-3, 1))
                                )
                            ),
                        }
                    )
        print("done", season, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUTS_DIR / "season_check.csv", index=False)
    return df


def write_report(df: pd.DataFrame) -> None:
    from plpredictor.evaluation.ml_eval import md_table

    by_event = df.groupby(["scale", "event"])[["brier", "log_loss"]].mean().reset_index()
    overall = df.groupby("scale")[["brier", "log_loss"]].mean().reset_index()
    text = f"""# Season-level check: does parameter uncertainty help?

For {len(SEASONS)} past seasons, simulated from after ~10 and ~20 games per team and scored the
title / top-4 / relegation probabilities against the real final tables (lower is better).
`scale` 0 = point estimate (no uncertainty), 1 = Laplace-approximation draws, >1 = wider.
Only {df.groupby(["season", "cutoff"]).ngroups} season/cut-off cases, so differences are small and noisy.

## Overall
{md_table(overall)}

## By event
{md_table(by_event)}
"""
    (config.REPORTS_DIR / "season_check.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    out = run()
    write_report(out)
    print(out.groupby("scale")[["brier", "log_loss"]].mean().round(4))
