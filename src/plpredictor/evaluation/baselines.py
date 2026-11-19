"""Baseline forecasters and the Phase 2 comparison table."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.evaluation import backtest, metrics
from plpredictor.models.elo import EloParams, EloProbModel, run_elo

CLASSES = ["H", "D", "A"]


def base_rate_fp(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical home/draw/away frequencies from E0 matches before the test window."""
    tr = train[train["div"] == "E0"]
    freq = tr["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    return np.tile(freq, (len(test), 1))


def market_preds(matches: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Margin-free market probabilities: 'pre' (pre-closing) or 'close'."""
    pre = "p_mkt" if kind == "pre" else "p_close"
    return matches[[f"{pre}_h", f"{pre}_d", f"{pre}_a"]].set_axis(["p_h", "p_d", "p_a"], axis=1)


def make_elo_fp(elo_frame: pd.DataFrame, params: EloParams):
    """fit_predict using precomputed pre-match Elo (each rating only used past results)."""

    def fp(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        tr = train[(train["div"] == "E0")].join(elo_frame[["elo_h", "elo_a"]])
        model = EloProbModel(params.home_adv).fit(
            tr["elo_h"], tr["elo_a"], metrics.outcome_index(tr["result"])
        )
        te = test.join(elo_frame[["elo_h", "elo_a"]])
        return model.predict(te["elo_h"], te["elo_a"])

    return fp


def tune_elo(matches: pd.DataFrame) -> tuple[EloParams, pd.DataFrame]:
    """Small grid search on the validation seasons only."""
    rows = []
    best, best_rps = EloParams(), np.inf
    for k in (10, 15, 20, 25, 30):
        for ha in (70, 90, 110):
            for reg in (0.2, 0.4):
                for gap in (150, 200, 300):
                    p = EloParams(k=k, home_adv=ha, regress=reg, league_gap=gap)
                    frame, _ = run_elo(matches, p)
                    preds = backtest.walk_forward(
                        make_elo_fp(frame.set_index(matches.index), p),
                        matches,
                        backtest.VALIDATION_SEASONS,
                        refit_days=365,
                    )
                    r = backtest.score(preds, matches)["rps"]
                    rows.append({"k": k, "home_adv": ha, "regress": reg, "gap": gap, "rps": r})
                    if r < best_rps:
                        best, best_rps = p, r
    return best, pd.DataFrame(rows).sort_values("rps")


def comparison_table(matches: pd.DataFrame, elo_params: EloParams) -> pd.DataFrame:
    """RPS / log loss / Brier / accuracy for every baseline on validation and test seasons."""
    elo_frame, _ = run_elo(matches, elo_params)
    elo_frame.index = matches.index
    rows = []
    for label, seasons in (
        ("validation 2016-21", backtest.VALIDATION_SEASONS),
        ("test 2021-27", backtest.TEST_SEASONS),
    ):
        base = backtest.walk_forward(base_rate_fp, matches, seasons, refit_days=365)
        elo = backtest.walk_forward(make_elo_fp(elo_frame, elo_params), matches, seasons, 365)
        pre = market_preds(matches, "pre").loc[base.index]
        close = market_preds(matches, "close").loc[base.index]
        common = pre.notna().all(axis=1) & close.notna().all(axis=1)  # fair comparison
        for name, p in (
            ("base rates", base),
            ("Elo", elo),
            ("market pre-closing", pre),
            ("market closing", close),
        ):
            rows.append({"period": label, "model": name, **backtest.score(p[common], matches)})
    return pd.DataFrame(rows)


def run() -> None:
    matches = build_dataset.load_matches()
    best, grid = tune_elo(matches)
    print("Best Elo params:", best)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS_DIR / "elo_best.json").write_text(json.dumps(asdict(best)))
    table = comparison_table(matches, best)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.OUTPUTS_DIR / "baselines.csv", index=False)
    grid.to_csv(config.OUTPUTS_DIR / "elo_grid.csv", index=False)
    print(table.round(4).to_string(index=False))


def load_best_elo() -> EloParams:
    """Tuned Elo parameters saved by ``run`` (defaults if not yet tuned)."""
    path = config.OUTPUTS_DIR / "elo_best.json"
    return EloParams(**json.loads(path.read_text())) if path.exists() else EloParams()


if __name__ == "__main__":
    run()
