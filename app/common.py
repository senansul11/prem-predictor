"""Shared helpers for the Streamlit app: data loading, labels, styling."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"
REPORTS = ROOT / "reports"

# Hosted environments (e.g. Streamlit Community Cloud) only install requirements.txt, so make the
# project's own code in src/ importable here. Locally it is already installed, so this is harmless.
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from plpredictor.data.team_names import short_name  # noqa: E402
from ui import CSS  # noqa: E402

DISCLAIMER = (
    "For entertainment and educational purposes only. Not betting advice. "
    "Predictions are based only on past results, so they cannot know about injuries, "
    "line-ups or transfers."
)
CREDIT = "Match data: [football-data.co.uk](https://www.football-data.co.uk). Fixtures: footballdata.io."
OUTCOME_TEXT = {"H": "Home win", "D": "Draw", "A": "Away win"}


@dataclass
class Data:
    meta: dict
    pred: pd.DataFrame  # remaining fixtures with predictions
    matrices: np.ndarray  # (fixtures, 11, 11) score probabilities, same order as pred
    table: pd.DataFrame  # current league table (index = team)
    summary: pd.DataFrame  # simulation summary (uncertainty-aware)
    positions: pd.DataFrame  # finishing position probabilities (index = team)
    ratings: pd.DataFrame  # attack / defence for current PL teams
    results: pd.DataFrame  # played matches this season
    params: dict  # global scoring parameters


@st.cache_data(show_spinner=False)
def _load(version: str) -> Data:  # `version` makes the cache reset after every data refresh
    meta = json.loads((OUT / "metadata.json").read_text())
    pred = pd.read_csv(OUT / "predictions.csv", parse_dates=["date"])
    mats = np.load(OUT / "fixture_matrices.npz")["matrices"].astype(float)
    table = pd.read_csv(OUT / "current_table.csv", index_col=0)
    summary = pd.read_csv(OUT / "simulation_summary.csv")
    positions = pd.read_csv(OUT / "position_distribution.csv", index_col=0)
    ratings = pd.read_csv(OUT / "team_ratings.csv")
    ratings = ratings[ratings["team"].isin(table.index)].reset_index(drop=True)
    results = pd.read_csv(OUT / "season_results.csv", parse_dates=["date"])
    params = json.loads((OUT / "model_params.json").read_text())
    return Data(meta, pred, mats, table, summary, positions, ratings, results, params)


def load() -> Data:
    version = json.loads((OUT / "metadata.json").read_text())["generated_at"]
    return _load(version)


def setup(title: str, icon: str = "⚽") -> Data:
    """Page config, shared header and data load. Call first on every page."""
    st.set_page_config(page_title=f"{title} · Prem Predictor", page_icon=icon, layout="wide")
    data = load()
    st.markdown(CSS, unsafe_allow_html=True)
    return data


def footer(data: Data) -> None:
    st.divider()
    st.caption(f"{DISCLAIMER}  \n{CREDIT}")
    st.caption(
        f"Data as of **{data.meta['data_as_of']}** (last result included) · "
        f"updated {data.meta['generated_at'][:16].replace('T', ' ')} UTC"
    )


def fixture_label(row) -> str:
    when = row["date"].strftime("%d %b %H:%M") if pd.notna(row["date"]) else "date TBC"
    return f"{short_name(row['home_team'])} v {short_name(row['away_team'])} ({when})"


def call_text(row) -> str:
    """Plain-words call: 'Arsenal win', 'Draw'."""
    if row["pred_outcome"] == "H":
        return f"{short_name(row['home_team'])} win"
    if row["pred_outcome"] == "A":
        return f"{short_name(row['away_team'])} win"
    return "Draw"


def zone_colour(position: int, n_teams: int = 20) -> str:
    """Background tint for table rows: Champions League, Europe, relegation."""
    if position <= 4:
        return "background-color: rgba(34,139,34,0.18)"
    if position == 5:
        return "background-color: rgba(30,144,255,0.15)"
    if position > n_teams - 3:
        return "background-color: rgba(220,20,60,0.18)"
    return ""


@st.cache_data(show_spinner=False)
def headline_accuracy() -> float | None:
    """Right-result rate of the final model on the test seasons (from the saved evaluation)."""
    path = OUT / "improve_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["period"] == "test 2021-27") & df["model"].str.startswith("FINAL")]
    return float(df["accuracy"].iloc[0]) if len(df) else None
