"""Fit the model, predict remaining fixtures, simulate the season and write outputs."""

from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.data.sources import fixtures_api
from plpredictor.models.dixon_coles import (
    DCConfig,
    DixonColes,
    expected_goals,
    matrix_summary,
    parameter_draws,
    predicted_pick,
    score_matrices,
    top_scorelines,
)
from plpredictor.simulation import season_sim

N_DRAWS = 200  # parameter draws; each simulates n_sims / N_DRAWS seasons


def run(n_sims: int = config.N_SIMS, uncertainty: bool = True) -> dict:
    matches = build_dataset.load_matches()
    played_all = matches[matches["status"] == "played"]
    season = config.season_label(config.current_season_start())
    cur = played_all[(played_all["season"] == season) & (played_all["div"] == config.LEAGUE)]
    teams = sorted(set(cur["home_team"]) | set(cur["away_team"]))
    assert len(teams) == 20, f"expected 20 teams, found {len(teams)}"

    cfg = DCConfig(**json.loads((config.OUTPUTS_DIR / "dc_config.json").read_text()))
    model = DixonColes(cfg).fit(played_all)
    fixtures = fixtures_api.full_remaining(cur, teams, fixtures_api.load_cached())
    mats = model.matrices(fixtures)

    table = season_sim.current_table(cur, teams)
    t0 = time.time()
    point = season_sim.simulate(table, fixtures, mats, n_sims=n_sims)
    if uncertainty:
        ref = played_all["date"].max() + pd.Timedelta(days=1)
        draws = parameter_draws(played_all, ref, cfg, model.fit_, k=N_DRAWS)
        per_draw = []
        for d in draws:
            lam, mu = expected_goals(d, fixtures["home_team"], fixtures["away_team"])
            per_draw.append(score_matrices(lam, mu, d.rho, cfg.max_goals))
        res = season_sim.simulate_uncertain(table, fixtures, per_draw, n_sims=n_sims)
    else:
        res = point
    elapsed = time.time() - t0

    now = table[["played", "pts", "gd"]].rename(columns={"pts": "pts_now", "gd": "gd_now"})
    summary = res.summary().merge(now, left_on="team", right_index=True)
    pred = pd.concat([fixtures, matrix_summary(mats)], axis=1)
    pred["top_scores"] = [
        ", ".join(f"{h}-{a} ({p:.0%})" for h, a, p in top_scorelines(m, 3)) for m in mats
    ]
    picks = pd.DataFrame([predicted_pick(m) for m in mats])
    pred["pred_outcome"] = picks["outcome"]
    pred["pred_score"] = picks["home_goals"].astype(str) + "-" + picks["away_goals"].astype(str)
    pred["pred_outcome_prob"] = picks["outcome_prob"]
    pred["pred_score_prob"] = picks["score_prob"]
    pred["confidence"] = picks["confidence"]
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = config.OUTPUTS_DIR / "fixture_matrices.npz"
    np.savez_compressed(npz_path, matrices=mats.astype("float32"))
    results_cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    cur[results_cols].to_csv(config.OUTPUTS_DIR / "season_results.csv", index=False)
    f = model.fit_
    params = {"home_adv": f.home_adv, "mu0": f.mu0, "rho": f.rho, "max_goals": cfg.max_goals}
    (config.OUTPUTS_DIR / "model_params.json").write_text(json.dumps(params, indent=2))
    summary.to_csv(config.OUTPUTS_DIR / "simulation_summary.csv", index=False)
    point.summary().to_csv(config.OUTPUTS_DIR / "simulation_summary_point.csv", index=False)
    res.position_distribution().to_csv(config.OUTPUTS_DIR / "position_distribution.csv")
    pred.to_csv(config.OUTPUTS_DIR / "predictions.csv", index=False)
    model.team_table().to_csv(config.OUTPUTS_DIR / "team_ratings.csv", index=False)
    table.to_csv(config.OUTPUTS_DIR / "current_table.csv")
    meta = {
        "data_as_of": str(cur["date"].max().date()),
        "season": season,
        "matches_played": int(len(cur)),
        "fixtures_remaining": int(len(fixtures)),
        "fixtures_with_dates": int(fixtures["dated"].sum()),
        "n_sims": n_sims,
        "parameter_uncertainty": uncertainty,
        "n_parameter_draws": N_DRAWS if uncertainty else 0,
        "seed": config.SEED,
        "sim_seconds": round(elapsed, 2),
        "model": "dixon-coles",
        "model_config": cfg.__dict__,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (config.OUTPUTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"meta": meta, "summary": summary, "table": table}


if __name__ == "__main__":
    out = run()
    print(out["meta"])
    print(out["table"].to_string())
    print(out["summary"].round(3).to_string(index=False))
