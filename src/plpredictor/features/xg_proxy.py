# ruff: noqa: E501
"""Shot-based expected-goals proxy (real xG exists only for the newest season).

Goals are partly luck, so a team's *chances* (shots, weighted by quality) describe its strength
more steadily. We estimate expected goals from shots on target and other shots by a no-intercept
least-squares fit, using only seasons up to ``FIT_UNTIL`` (so test seasons never leak in).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from plpredictor import config

FIT_UNTIL = "2020-21"
COEF_PATH = config.OUTPUTS_DIR / "xg_proxy.json"


def fit_proxy(matches: pd.DataFrame, until: str = FIT_UNTIL) -> dict[str, float]:
    """Return {'sot': a, 'off': b}: goals ~ a*SOT + b*(shots - SOT)."""
    m = matches[(matches["status"] == "played") & (matches["season"] <= until)]
    m = m.dropna(subset=["home_shots", "away_shots", "home_sot", "away_sot"])
    sot = np.r_[m["home_sot"], m["away_sot"]]
    off = np.r_[m["home_shots"] - m["home_sot"], m["away_shots"] - m["away_sot"]]
    goals = np.r_[m["home_goals"], m["away_goals"]]
    coef, *_ = np.linalg.lstsq(np.column_stack([sot, off]), goals, rcond=None)
    return {"sot": float(coef[0]), "off": float(coef[1])}


def add_proxy(matches: pd.DataFrame, coef: dict[str, float]) -> pd.DataFrame:
    """Add ``home_xgp`` / ``away_xgp`` columns (NaN where shot data is missing)."""
    out = matches.copy()
    out["home_xgp"] = coef["sot"] * out["home_sot"] + coef["off"] * (out["home_shots"] - out["home_sot"])
    out["away_xgp"] = coef["sot"] * out["away_sot"] + coef["off"] * (out["away_shots"] - out["away_sot"])
    return out


def load_or_fit(matches: pd.DataFrame) -> dict[str, float]:
    if COEF_PATH.exists():
        return json.loads(COEF_PATH.read_text())
    coef = fit_proxy(matches)
    COEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    COEF_PATH.write_text(json.dumps(coef))
    return coef
