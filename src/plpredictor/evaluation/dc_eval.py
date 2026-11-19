"""Tune and evaluate the Dixon-Coles model (Phase 3).

Run: python -m plpredictor.evaluation.dc_eval
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.evaluation import backtest, baselines
from plpredictor.features.momentum import compute_momentum
from plpredictor.models.dixon_coles import DCConfig, DixonColes

MOMENTUM_MARGIN = 0.0003  # min validation RPS gain to keep the momentum term


def make_dc_fp(cfg: DCConfig):
    """fit_predict for the walk-forward backtest (warm-starts from the previous fit)."""
    model = DixonColes(cfg)

    def fp(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model.fit(train, ref_date=test["date"].min())
        return model.predict(test)[["p_h", "p_d", "p_a"]].to_numpy()

    return fp


def val_rps(matches: pd.DataFrame, cfg: DCConfig, refit_days: int = 28) -> float:
    preds = backtest.walk_forward(
        make_dc_fp(cfg), matches, backtest.VALIDATION_SEASONS, refit_days=refit_days
    )
    return backtest.score(preds, matches)["rps"]


def tune(matches: pd.DataFrame) -> tuple[DCConfig, pd.DataFrame]:
    """Coordinate descent on validation seasons: one hyper-parameter at a time."""
    cfg = DCConfig()
    log = []
    grids = [
        ("xi", [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]),
        ("ridge", [0.05, 0.15, 0.3, 0.6, 1.2, 3.0]),
        ("season_decay", [1.0, 0.9, 0.8, 0.7]),
        ("w_e1", [0.0, 0.25, 0.5, 1.0]),
        ("red_w", [1.0, 0.5]),
        ("use_momentum", [False, True]),
    ]
    for name, values in grids:
        best_v, best_r = getattr(cfg, name), np.inf
        for v in values:
            r = val_rps(matches, replace(cfg, **{name: v}))
            log.append({"param": name, "value": v, "val_rps": r})
            print(f"  {name}={v}: val RPS {r:.5f}", flush=True)
            # momentum must beat "off" by a real margin (not noise) to be kept
            margin = MOMENTUM_MARGIN if (name == "use_momentum" and v) else 0.0
            if r < best_r - margin or best_r == np.inf:
                best_v, best_r = v, r
        cfg = replace(cfg, **{name: best_v})
        print(f"-> {name} = {best_v}", flush=True)
    return cfg, pd.DataFrame(log)


def sanity_checks(matches: pd.DataFrame, cfg: DCConfig) -> dict[str, float]:
    """Model vs market: P(over 2.5) and expected goal difference vs the Asian handicap."""
    played = matches[matches["status"] == "played"]
    rows = []
    for season in backtest.TEST_SEASONS:
        test = played[(played["season"] == season) & (played["div"] == "E0")]
        for chunk in backtest.chunk_by_days(test, 28):
            train = played[played["date"] < chunk["date"].min()]
            model = DixonColes(cfg).fit(train, ref_date=chunk["date"].min())
            pr = model.predict(chunk)
            mk = chunk[["p_over25", "ah_line", "home_goals", "away_goals"]]
            rows.append(pd.concat([pr, mk.rename(columns={"p_over25": "mkt_over25"})], axis=1))
    d = pd.concat(rows).dropna(subset=["mkt_over25", "ah_line"])
    exp_gd = d["xg_h"] - d["xg_a"]
    return {
        "n": len(d),
        "model_mean_total_goals": float((d["xg_h"] + d["xg_a"]).mean()),
        "actual_mean_total_goals": float((d["home_goals"] + d["away_goals"]).mean()),
        "model_mean_p_over25": float(d["p_over25"].mean()),
        "market_mean_p_over25": float(d["mkt_over25"].mean()),
        "corr_over25_model_vs_market": float(np.corrcoef(d["p_over25"], d["mkt_over25"])[0, 1]),
        "corr_gd_model_vs_ah": float(np.corrcoef(exp_gd, -d["ah_line"])[0, 1]),
        "mean_gd_model": float(exp_gd.mean()),
        "mean_gd_market": float((-d["ah_line"]).mean()),
        "mean_gd_actual": float((d["home_goals"] - d["away_goals"]).mean()),
    }


def run(tune_params: bool = True) -> None:
    matches = build_dataset.load_matches()
    elo_params = baselines.load_best_elo()
    matches = matches.join(compute_momentum(matches, elo_params))
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = config.OUTPUTS_DIR / "dc_config.json"
    if tune_params or not cfg_path.exists():
        cfg, log = tune(matches)
        log.to_csv(config.OUTPUTS_DIR / "dc_tuning.csv", index=False)
        cfg_path.write_text(json.dumps(asdict(cfg)))
    cfg = DCConfig(**json.loads(cfg_path.read_text()))
    print("Tuned config:", cfg)

    table = baselines.comparison_table(matches, elo_params)
    variants = {
        "Dixon-Coles (tuned)": cfg,
        "DC plain (time decay only)": DCConfig(xi=cfg.xi, ridge=cfg.ridge, w_e1=cfg.w_e1),
        "DC tuned + momentum": replace(cfg, use_momentum=True),
        "DC tuned, no momentum": replace(cfg, use_momentum=False),
    }
    rows = []
    for label, seasons in (
        ("validation 2016-21", backtest.VALIDATION_SEASONS),
        ("test 2021-27", backtest.TEST_SEASONS),
    ):
        mk_pre = baselines.market_preds(matches, "pre")
        for name, vcfg in variants.items():
            p = backtest.walk_forward(make_dc_fp(vcfg), matches, seasons, refit_days=7)
            common = mk_pre.loc[p.index].notna().all(axis=1)
            rows.append({"period": label, "model": name, **backtest.score(p[common], matches)})
            print(rows[-1], flush=True)
    out = pd.concat([table, pd.DataFrame(rows)]).sort_values(["period", "rps"])
    out.to_csv(config.OUTPUTS_DIR / "model_comparison_dc.csv", index=False)
    print(out.round(4).to_string(index=False))
    checks = sanity_checks(matches, cfg)
    (config.OUTPUTS_DIR / "dc_sanity.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    run()
