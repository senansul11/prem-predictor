import numpy as np
import pandas as pd
import pytest

from plpredictor.evaluation import backtest, metrics
from plpredictor.models import elo


def test_rps_perfect_and_worst():
    y = np.array([0, 2])
    perfect = np.array([[1, 0, 0], [0, 0, 1.0]])
    assert metrics.rps(perfect, y) == 0.0
    worst = np.array([[0, 0, 1.0], [1.0, 0, 0]])
    assert metrics.rps(worst, y) == pytest.approx(1.0)


def test_rps_ordinal_known_value():
    # forecast .5/.25/.25, home win: cum diffs (.5, .25) -> (.25 + .0625)/2
    assert metrics.rps(np.array([[0.5, 0.25, 0.25]]), np.array([0])) == pytest.approx(0.15625)
    # predicting a draw when away wins is less wrong than predicting a home win
    p_draw = metrics.rps(np.array([[0, 1.0, 0]]), np.array([2]))
    p_home = metrics.rps(np.array([[1.0, 0, 0]]), np.array([2]))
    assert p_draw < p_home


def test_logloss_brier_accuracy():
    p = np.array([[0.6, 0.3, 0.1]])
    assert metrics.log_loss(p, np.array([0])) == pytest.approx(-np.log(0.6))
    assert metrics.brier(p, np.array([0])) == pytest.approx(0.4**2 + 0.3**2 + 0.1**2)
    assert metrics.accuracy(p, np.array([0])) == 1.0


def test_elo_update_zero_sum_and_direction():
    p = elo.EloParams()
    h, a = elo.update(1500, 1500, 2, 0, p)
    assert h + a == pytest.approx(3000)
    assert h > 1500 > a
    h2, a2 = elo.update(1500, 1500, 0, 0, p)  # home advantage -> draw hurts home side
    assert h2 < 1500 < a2


def _toy():
    rows = []
    d = pd.Timestamp("2020-08-20")
    for i in range(6):
        rows.append(("E0", "2020-21", d + pd.Timedelta(days=7 * i), "A", "B", 2, 0, "H"))
    df = pd.DataFrame(
        rows, columns=["div", "season", "date", "home_team", "away_team",
                       "home_goals", "away_goals", "result"]
    )  # fmt: skip
    df["status"] = "played"
    df["match_id"] = range(len(df))
    return df


def test_elo_no_future_leakage():
    m = _toy()
    full, _ = elo.run_elo(m)
    trunc, _ = elo.run_elo(m.iloc[:3])
    # pre-match ratings of the first 3 matches do not depend on later matches
    assert np.allclose(full["elo_h"].iloc[:3], trunc["elo_h"].iloc[:3])
    assert full["elo_h"].iloc[0] == full["elo_h"].iloc[0]
    assert full["elo_h"].iloc[5] > full["elo_h"].iloc[0]  # A keeps winning


def test_walk_forward_trains_only_on_past():
    m = _toy()
    seen = []

    def fp(train, test):
        assert (train["date"] < test["date"].min()).all()
        seen.append(len(train))
        return np.full((len(test), 3), 1 / 3)

    out = backtest.walk_forward(fp, m, ["2020-21"], refit_days=7)
    assert len(out) == 6 and seen == [0, 1, 2, 3, 4, 5]
