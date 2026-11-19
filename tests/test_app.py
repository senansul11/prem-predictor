"""Headless checks of the Streamlit pages (they read data/outputs, so skip if absent)."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "outputs" / "fixture_matrices.npz").exists(), reason="run simulate first"
)

PAGES = [
    "app/Home.py",
    "app/pages/1_Match_Predictor.py",
    "app/pages/2_What_If_Simulator.py",
    "app/pages/3_Team_Page.py",
    "app/pages/4_How_Accurate.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_errors(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_match_predictor_changes_with_teams():
    at = AppTest.from_file(str(ROOT / "app/pages/1_Match_Predictor.py"), default_timeout=60).run()
    first = [m.value for m in at.metric]
    at.selectbox[0].select("Liverpool").run()
    at.selectbox[1].select("Manchester City").run()
    assert not at.exception
    assert [m.value for m in at.metric] != first


def test_what_if_lock_changes_title_odds(monkeypatch):
    # The hosted app has no matplotlib: make any import of it fail, as it would online.
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", None)
    page = str(ROOT / "app/pages/2_What_If_Simulator.py")
    at = AppTest.from_file(page, default_timeout=120).run()
    pred = pd.read_csv(ROOT / "data/outputs/predictions.csv", parse_dates=["date"])
    idx = pred[(pred.home_team == "Arsenal") & (pred.away_team == "Fulham")].index[0]
    at.multiselect[0].select(int(idx)).run()
    at.number_input[0].set_value(0)  # Arsenal goals
    at.number_input[1].set_value(3)  # Fulham goals
    at.radio[0].set_value(1000)
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    before, after, key = at.session_state["whatif"]
    assert key  # a result was locked
    assert after.loc["Arsenal", "p_title"] < before.loc["Arsenal", "p_title"]
    assert after.loc["Arsenal", "exp_points"] < before.loc["Arsenal", "exp_points"]
