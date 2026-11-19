# ruff: noqa: E501
"""Phase 3b experiments: shot-based xG proxy and team-specific home advantage.

Tuning uses validation seasons only (2016-21); the test seasons (2021-27) are scored once at
the end. Run: python -m plpredictor.evaluation.improve_eval
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.evaluation import backtest, baselines
from plpredictor.evaluation.dc_eval import make_dc_fp, val_rps
from plpredictor.evaluation.ml_eval import md_table
from plpredictor.features import xg_proxy
from plpredictor.models.dixon_coles import DCConfig, DixonColes

MARGIN = 0.0003  # an extra feature must beat "off" by this much on validation to be kept
GRIDS = [
    ("xg_blend", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("team_home_ridge", [0.0, 100.0, 50.0, 25.0, 10.0, 5.0]),
    ("xi", [0.001, 0.0015, 0.002, 0.003]),
    ("ridge", [0.05, 0.15, 0.3]),
]


def tune(matches: pd.DataFrame, start: DCConfig):
    cfg, log = start, []
    for name, values in GRIDS:
        best_v, best_r = getattr(cfg, name), val_rps(matches, cfg)
        log.append({"param": name, "value": f"{best_v} (current)", "val_rps": best_r})
        for v in values:
            if v == getattr(cfg, name):
                continue
            r = val_rps(matches, replace(cfg, **{name: v}))
            log.append({"param": name, "value": v, "val_rps": r})
            print(f"  {name}={v}: {r:.5f}", flush=True)
            margin = MARGIN if name == "team_home_ridge" and best_v == 0.0 else 1e-5
            if r < best_r - margin:
                best_v, best_r = v, r
        cfg = replace(cfg, **{name: best_v})
        print(f"-> {name} = {best_v}  (val RPS {best_r:.5f})", flush=True)
    return cfg, pd.DataFrame(log)


def total_goal_bias(matches: pd.DataFrame, cfg: DCConfig, seasons: list[str]) -> float:
    played = matches[matches["status"] == "played"]
    diffs = []
    for season in seasons:
        test = played[(played["season"] == season) & (played["div"] == config.LEAGUE)]
        for chunk in backtest.chunk_by_days(test, 28):
            train = played[played["date"] < chunk["date"].min()]
            model = DixonColes(cfg).fit(train, ref_date=chunk["date"].min())
            pr = model.predict(chunk)
            diffs.append((pr["xg_h"] + pr["xg_a"]) - (chunk["home_goals"] + chunk["away_goals"]))
    return float(pd.concat(diffs).mean())


def home_advantage_table(matches: pd.DataFrame, cfg: DCConfig) -> pd.DataFrame:
    """Learned home advantage for the current Premier League clubs vs raw recent history."""
    played = matches[matches["status"] == "played"]
    model = DixonColes(cfg).fit(played)
    f = model.fit_
    cur = played[(played["season"] == "2026-27") & (played["div"] == config.LEAGUE)]
    teams = sorted(set(cur["home_team"]) | set(cur["away_team"]))
    hist = played[(played["div"] == config.LEAGUE) & (played["season"] >= "2016-17")]
    rows = []
    for t in teams:
        hh = hist[hist["home_team"] == t]
        aa = hist[hist["away_team"] == t]
        ppg_h = (3 * (hh["result"] == "H") + (hh["result"] == "D")).mean() if len(hh) else np.nan
        ppg_a = (3 * (aa["result"] == "A") + (aa["result"] == "D")).mean() if len(aa) else np.nan
        rows.append(
            {
                "team": t,
                "home_goal_boost_%": 100 * (np.exp(f.team_home[f.index[t]]) - 1) if f.team_home is not None else 0.0,
                "home_ppg": ppg_h,
                "away_ppg": ppg_a,
                "home_minus_away_ppg": ppg_h - ppg_a,
                "home_matches_in_history": len(hh),
            }
        )
    out = pd.DataFrame(rows).sort_values("home_goal_boost_%", ascending=False).reset_index(drop=True)
    return out


def run(tune_params: bool = True) -> None:
    matches = build_dataset.load_matches()
    coef = xg_proxy.load_or_fit(matches)
    matches = xg_proxy.add_proxy(matches, coef)
    base_path = config.OUTPUTS_DIR / "dc_config.json"
    v1_path = config.OUTPUTS_DIR / "dc_config_v1.json"
    if not v1_path.exists():
        shutil.copy(base_path, v1_path)
    base = DCConfig(**json.loads(v1_path.read_text()))

    cfg_path = config.OUTPUTS_DIR / "dc_config_v2.json"
    if tune_params or not cfg_path.exists():
        best, log = tune(matches, base)
        log.to_csv(config.OUTPUTS_DIR / "improve_tuning.csv", index=False)
        cfg_path.write_text(json.dumps(asdict(best)))
    best = DCConfig(**json.loads(cfg_path.read_text()))
    print("Best:", best)

    variants = {
        "Baseline Dixon-Coles (v1)": base,
        "+ shot-based xG proxy only": replace(base, xg_blend=best.xg_blend),
        "+ team home advantage only (ridge 10)": replace(base, team_home_ridge=10.0),
        "+ xG proxy and team home (ridge 10)": replace(base, xg_blend=best.xg_blend, team_home_ridge=10.0),
        "FINAL: xG proxy + retuned settings": best,
    }
    mk_pre = baselines.market_preds(matches, "pre")
    rows = []
    for label, seasons in (
        ("validation 2016-21", backtest.VALIDATION_SEASONS),
        ("test 2021-27", backtest.TEST_SEASONS),
    ):
        for name, vcfg in variants.items():
            p = backtest.walk_forward(make_dc_fp(vcfg), matches, seasons, refit_days=7)
            common = mk_pre.loc[p.index].notna().all(axis=1)
            sc = backtest.score(p[common], matches)
            rows.append({"period": label, "model": name, **sc})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(config.OUTPUTS_DIR / "improve_comparison.csv", index=False)
    bias = {
        "baseline": total_goal_bias(matches, base, backtest.TEST_SEASONS),
        "final": total_goal_bias(matches, best, backtest.TEST_SEASONS),
    }
    hat = home_advantage_table(matches, replace(best, team_home_ridge=10.0))
    hat.to_csv(config.OUTPUTS_DIR / "home_advantage.csv", index=False)
    write_report(table, pd.read_csv(config.OUTPUTS_DIR / "improve_tuning.csv"), bias, hat, best, coef)


def write_report(table, tuning, bias, hat, best, coef) -> None:
    test = table[table["period"] == "test 2021-27"].reset_index(drop=True)
    gain = test["rps"].iloc[0] - test["rps"].iloc[-1]
    text = f"""# Model improvement experiments

Four ideas were tested on the Dixon-Coles goals model, one at a time, with the usual rules:
tune on validation seasons 2016-21 only, then score on test seasons 2021-27 (walk-forward,
weekly refits). RPS: lower is better.

## Result
{md_table(table)}

RPS improvement on test (baseline minus final): **{gain:+.4f}**.
Average error in predicted total goals per match (model - actual): baseline {bias['baseline']:+.3f}, final {bias['final']:+.3f}.

Final settings: `{json.dumps(asdict(best))}`

## Ideas 2 and 3 (draws, goal level): no change made
The diagnosis showed draw probabilities and the goal level were already well calibrated
(draws: 27.6% predicted vs 26.3% actual in the closest games). Post-hoc corrections made test RPS
slightly *worse* (baseline 0.20124): outcome recalibration 0.20155, draw scaling 0.20132,
trailing goal-level rescaling 0.20173-0.20211. The one big miss is 2023-24, an unusually
high-scoring season (3.28 goals/game), which no stable model predicts.

## Idea 1: shot-based xG proxy
Goals ~ {coef['sot']:.3f} x shots on target + {coef['off']:.3f} x other shots (fitted on seasons up to 2020-21).
Each simulated "target" mixes real goals with this proxy; chosen mix: **{best.xg_blend:.0%} proxy**.

## Idea 4: team-specific home advantage (current clubs)
Learned from historic results (fitted with ridge 10 for display), shrunk towards the league average. **Not used in the final model**: it did not beat the plain model on validation by the required margin, and was slightly worse on test.
`home_goal_boost_%` is the extra boost to the team's home goal rate vs the average club.
{md_table(hat, 3)}

## Tuning log (validation RPS)
{md_table(tuning, 5)}
"""
    (config.REPORTS_DIR / "improvement_experiments.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
