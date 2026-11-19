# ruff: noqa: E501
"""Phase 5 evaluation: features, XGBoost variants, ablations, blend, and the report.

Run: python -m plpredictor.evaluation.ml_eval
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.evaluation import backtest, baselines, metrics
from plpredictor.features import build_features as bf
from plpredictor.models.dixon_coles import DCConfig
from plpredictor.models.ensemble import blend
from plpredictor.models.xgb_model import XGBModel

FEATURES_PATH = config.PROCESSED_DIR / "features.parquet"
SEASONS_ALL = ["2019-20", "2020-21", *backtest.TEST_SEASONS]
VAL = ["2019-20", "2020-21"]
BASE = bf.form_features() + bf.ELO_FEATS + bf.MOM_FEATS + bf.DC_FEATS

VARIANTS = {
    "XGB (all features, season-weighted, calibrated)": dict(feats=BASE),
    "XGB raw (no calibration)": dict(feats=BASE, calibrate=False),
    "XGB unweighted seasons": dict(feats=BASE, decay=1.0),
    "XGB without momentum": dict(feats=[f for f in BASE if f not in bf.MOM_FEATS]),
    "XGB without Elo": dict(feats=[f for f in BASE if f not in bf.ELO_FEATS]),
    "XGB without Dixon-Coles features": dict(feats=[f for f in BASE if f not in bf.DC_FEATS]),
    "XGB + pre-closing odds": dict(feats=BASE + bf.ODDS_FEATS),
}


def load_features(matches: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    if FEATURES_PATH.exists() and not rebuild:
        return pd.read_parquet(FEATURES_PATH)
    elo = baselines.load_best_elo()
    dc_cfg = DCConfig(**json.loads((config.OUTPUTS_DIR / "dc_config.json").read_text()))
    feats = bf.build_features(matches, elo, dc_cfg)
    feats.to_parquet(FEATURES_PATH)
    return feats


def make_fp(spec: dict):
    def fp(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        tr = train[(train["div"] == config.LEAGUE) & train["dc_lam_h"].notna()]
        model = XGBModel(spec["feats"], spec.get("decay", 0.8), spec.get("calibrate", True))
        model.fit(tr, ref_date=test["date"].min())
        return model.predict(test)

    return fp


def md_table(df: pd.DataFrame, floats: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run() -> None:
    matches = build_dataset.load_matches()
    feats = load_features(matches)
    m = matches.join(feats)
    mk_pre = baselines.market_preds(matches, "pre")
    mk_close = baselines.market_preds(matches, "close")

    preds: dict[str, pd.DataFrame] = {}
    for name, spec in VARIANTS.items():
        print("running", name, flush=True)
        cache = config.PROCESSED_DIR / f"mlpred_{list(VARIANTS).index(name)}.parquet"
        if cache.exists():
            preds[name] = pd.read_parquet(cache)
            continue
        preds[name] = backtest.walk_forward(make_fp(spec), m, SEASONS_ALL, refit_days=42)
        preds[name].to_parquet(cache)

    base_idx = preds[next(iter(VARIANTS))].index
    dc = m.loc[base_idx, ["dc_p_h", "dc_p_d", "dc_p_a"]].set_axis(["p_h", "p_d", "p_a"], axis=1)
    elo_frame = m[["elo_h", "elo_a"]]
    elo_fp = baselines.make_elo_fp(elo_frame, baselines.load_best_elo())
    elo = backtest.walk_forward(elo_fp, matches, SEASONS_ALL, refit_days=365)

    # blend weight chosen on validation seasons only
    x = preds[next(iter(VARIANTS))]
    val_idx = m.loc[base_idx][m.loc[base_idx, "season"].isin(VAL)].index
    grid = np.round(np.linspace(0, 1, 11), 2)
    val_rps = {
        w: backtest.score(
            pd.DataFrame(blend(dc.loc[val_idx].to_numpy(), x.loc[val_idx].to_numpy(), w),
                         index=val_idx, columns=["p_h", "p_d", "p_a"]), m)["rps"]
        for w in grid
    }  # fmt: skip
    w_best = min(val_rps, key=val_rps.get)
    print("blend weight (XGB share) chosen on validation:", w_best, val_rps)

    named = {
        "Elo": elo, "Dixon-Coles (walk-forward)": dc, **preds,
        f"Blend DC+XGB (w_xgb={w_best})": pd.DataFrame(
            blend(dc.to_numpy(), x.to_numpy(), w_best), index=base_idx, columns=["p_h", "p_d", "p_a"]
        ),
        "Market pre-closing": mk_pre.loc[base_idx],
        "Market closing": mk_close.loc[base_idx],
    }  # fmt: skip
    test_idx = m.loc[base_idx][m.loc[base_idx, "season"].isin(backtest.TEST_SEASONS)].index
    common = test_idx[
        mk_pre.loc[test_idx].notna().all(axis=1) & mk_close.loc[test_idx].notna().all(axis=1)
    ]
    rows = [{"model": k, **backtest.score(v.loc[common], m)} for k, v in named.items()]
    table = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    sweep = pd.DataFrame(
        [{"w_xgb": w,
          "val_rps": val_rps[w],
          "test_rps": backtest.score(
              pd.DataFrame(blend(dc.loc[common].to_numpy(), x.loc[common].to_numpy(), w),
                           index=common, columns=["p_h", "p_d", "p_a"]), m)["rps"]}
         for w in grid]
    )  # fmt: skip
    per_season = pd.DataFrame(
        [
            {"season": s, **{k: backtest.score(named[k].loc[common[m.loc[common, "season"] == s]], m)["rps"]
                              for k in ("Dixon-Coles (walk-forward)", "Market closing", "Elo")}}
            for s in backtest.TEST_SEASONS
        ]
    )  # fmt: skip

    # final feature importance (fit on everything played before today)
    tr = m[(m["div"] == config.LEAGUE) & (m["status"] == "played") & m["dc_lam_h"].notna()]
    imp = XGBModel(BASE, 0.8, True).fit(tr, tr["date"].max() + pd.Timedelta(days=1)).importance()

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)
    sweep.to_csv(config.OUTPUTS_DIR / "blend_sweep.csv", index=False)
    imp.to_csv(config.OUTPUTS_DIR / "feature_importance.csv")
    write_report(table, sweep, per_season, imp, named, common, m, w_best)
    print(table.round(4).to_string(index=False))


def write_report(table, sweep, per_season, imp, named, common, m, w_best) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = metrics.outcome_index(m.loc[common, "result"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    cal_rows = []
    for label in ("Dixon-Coles (walk-forward)", f"Blend DC+XGB (w_xgb={w_best})", "Market closing"):
        cb = metrics.calibration_bins(named[label].loc[common].to_numpy(), y, outcome=0, bins=8)
        ax.plot(cb["mean_pred"], cb["observed"], "o-", label=label)
        cal_rows.append(cb.assign(model=label))
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed frequency")
    ax.legend(fontsize=7)
    fig.tight_layout()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.REPORTS_DIR / "calibration.png", dpi=120)

    cov = pd.read_csv(config.PROCESSED_DIR / "data_coverage.csv")
    cov = cov[cov["div"] == "E0"].drop(columns="div")
    dc_san = json.loads((config.OUTPUTS_DIR / "dc_sanity.json").read_text())
    tuning = pd.read_csv(config.OUTPUTS_DIR / "dc_tuning.csv")
    cfg = json.loads((config.OUTPUTS_DIR / "dc_config.json").read_text())
    text = f"""# Model evaluation

Generated by `python -m plpredictor.evaluation.ml_eval`. Lower RPS / log loss / Brier is better.
All numbers are **walk-forward** (each match predicted by a model that only saw earlier matches).
Hyper-parameters were tuned on validation seasons 2016-17 to 2020-21 only. The test seasons are
2021-22 to 2026-27 (the current season is partial: {int((m.loc[common, "season"] == "2026-27").sum())} matches).
Rows are restricted to matches that have both pre-closing and closing market odds so every
model is scored on identical matches (n = {len(common)}).

## Headline comparison (test seasons)
{md_table(table)}

Reading it: the **closing market** is the gold standard. The gap between our best model and the
market is the honest measure of what the model is missing (injuries, line-ups, news).

## RPS per test season
{md_table(per_season)}

## Blend weight (XGBoost share), chosen on validation seasons 2019-21
{md_table(sweep)}

Chosen weight: **{w_best}**. The test column is shown for transparency only; it was not used to choose.

## Dixon-Coles tuned settings
`{json.dumps(cfg)}`

Tuning log (validation RPS; one parameter at a time, see `data/outputs/dc_tuning.csv`):
{md_table(tuning, 5)}

## Sanity checks of the goals model vs the market (test seasons)
{md_table(pd.DataFrame([dc_san]).T.reset_index().rename(columns={"index": "check", 0: "value"}))}

## Feature importance (XGBoost, gain-based, fitted on all data to date)
{md_table(imp.head(15).reset_index().rename(columns={"index": "feature", 0: "importance"}))}

## Calibration (P(home win))
![calibration](calibration.png)

## Data coverage (Premier League)
{md_table(cov, 1)}

Notes: no `xG` before 2026-27, so v1 uses goals only. Pinnacle odds thin out during 2025-26 and
are missing in 2026-27, where the market average is used instead.
"""
    (config.REPORTS_DIR / "model_evaluation.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
