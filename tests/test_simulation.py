import numpy as np
import pandas as pd
import pytest

from plpredictor.models.dixon_coles import score_matrices
from plpredictor.simulation import season_sim
from plpredictor.simulation.tiebreakers import rank_positions

TEAMS = ["A", "B", "C", "D"]


def test_tiebreak_order_points_then_gd_then_goals():
    rng = np.random.default_rng(0)
    pts = np.array([[10, 10, 10, 9]], dtype=float)
    gd = np.array([[3, 5, 5, 20]], dtype=float)
    gf = np.array([[10, 8, 12, 30]], dtype=float)
    order = rank_positions(pts, gd, gf, rng)[0]
    assert order.tolist() == [2, 1, 0, 3]  # same pts: GD 5 beats 3; GF 12 beats 8; 9 pts last


def _setup(n_fix=12):
    played = pd.DataFrame(
        {"home_team": ["A", "C"], "away_team": ["B", "D"], "home_goals": [2, 0],
         "away_goals": [1, 0]}  # fmt: skip
    )
    table = season_sim.current_table(played, TEAMS)
    fx = season_sim.remaining_fixtures(played, TEAMS)
    lam = np.full(len(fx), 1.4)
    mats = score_matrices(lam, np.full(len(fx), 1.1), rho=-0.05)
    return table, fx, mats


def test_current_table_and_remaining():
    table, fx, _ = _setup()
    assert table.loc["A", "pts"] == 3 and table.loc["B", "pts"] == 0
    assert table.loc["C", "pts"] == table.loc["D", "pts"] == 1
    assert len(fx) == 12 - 2  # 4 teams double round robin = 12 games, 2 played


def test_sim_distributions_valid_and_deterministic():
    table, fx, mats = _setup()
    r1 = season_sim.simulate(table, fx, mats, n_sims=2000, seed=7)
    r2 = season_sim.simulate(table, fx, mats, n_sims=2000, seed=7)
    assert np.array_equal(r1.position_counts, r2.position_counts)
    assert np.allclose(r1.position_counts.sum(axis=0), 2000)  # each position filled every sim
    assert np.allclose(r1.position_counts.sum(axis=1), 2000)  # each team placed every sim
    s = r1.summary(europe_places=2, relegation_places=1)
    assert s["p_title"].sum() == pytest.approx(1.0)
    assert s["p_relegation"].sum() == pytest.approx(1.0)


def test_locked_result_changes_outcome():
    table, fx, mats = _setup()
    base = season_sim.simulate(table, fx, mats, n_sims=3000, seed=1).summary()
    # lock every remaining fixture involving A as a big A win
    locked = {}
    for i, r in enumerate(fx.itertuples()):
        if r.home_team == "A":
            locked[i] = (5, 0)
        elif r.away_team == "A":
            locked[i] = (0, 5)
    after = season_sim.simulate(table, fx, mats, n_sims=3000, seed=1, locked=locked).summary()
    pa = lambda s: s.set_index("team").loc["A", "p_title"]  # noqa: E731
    assert pa(after) > pa(base)
    assert pa(after) == pytest.approx(1.0)


def test_10k_sims_fast_for_full_league():
    import time

    teams = [f"T{i}" for i in range(20)]
    fx = pd.DataFrame([(h, a) for h in teams for a in teams if h != a],
                      columns=["home_team", "away_team"])  # fmt: skip
    table = season_sim.current_table(fx.iloc[:0].assign(home_goals=0, away_goals=0), teams)
    mats = score_matrices(np.full(len(fx), 1.5), np.full(len(fx), 1.2), -0.05)
    t = time.time()
    season_sim.simulate(table, fx, mats, n_sims=10_000)
    assert time.time() - t < 10
