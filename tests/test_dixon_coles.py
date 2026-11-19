import numpy as np
import pandas as pd
import pytest

from plpredictor.data import build_dataset
from plpredictor.features import momentum
from plpredictor.models import dixon_coles as dc


def _simulate(n_matches=3000, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    att = np.array([0.4, 0.2, 0.1, 0.0, -0.1, -0.2, -0.2, -0.2])
    dfn = np.array([0.3, 0.2, 0.1, 0.0, 0.0, -0.1, -0.2, -0.3])
    rows = []
    d0 = pd.Timestamp("2022-08-01")
    for k in range(n_matches):
        h, a = rng.choice(8, 2, replace=False)
        lam = np.exp(0.2 + 0.25 + att[h] - dfn[a])
        mu = np.exp(0.2 + att[a] - dfn[h])
        rows.append(("E0", "2022-23", d0 + pd.Timedelta(days=k % 300), teams[h], teams[a],
                     rng.poisson(lam), rng.poisson(mu)))  # fmt: skip
    df = pd.DataFrame(rows, columns=["div", "season", "date", "home_team", "away_team",
                                     "home_goals", "away_goals"])  # fmt: skip
    df["status"] = "played"
    df["home_red"] = df["away_red"] = 0.0
    return df, att + dfn


def test_parameter_recovery():
    df, true_overall = _simulate()
    cfg = dc.DCConfig(xi=0.0, ridge=0.01, w_e1=1.0)
    fit = dc.fit_dc(df, pd.Timestamp("2023-06-01"), cfg)
    overall = fit.attack + fit.defence
    assert np.corrcoef(overall, true_overall)[0, 1] > 0.95
    assert fit.home_adv == pytest.approx(0.25, abs=0.08)


def test_matrices_sum_to_one_and_wdl():
    lam, mu = np.array([1.5, 0.8, 2.6]), np.array([1.1, 1.0, 0.4])
    m = dc.score_matrices(lam, mu, rho=-0.05)
    assert m.shape == (3, 11, 11)
    assert np.allclose(m.sum(axis=(1, 2)), 1.0)
    s = dc.matrix_summary(m)
    assert np.allclose(s[["p_h", "p_d", "p_a"]].sum(axis=1), 1.0)
    assert s["p_h"].iloc[2] > s["p_a"].iloc[2]
    assert dc.top_scorelines(m[0], 3)[0][2] >= dc.top_scorelines(m[0], 3)[1][2]


def test_low_score_correction_direction():
    lam, mu = np.array([1.3]), np.array([1.1])
    base = dc.score_matrices(lam, mu, rho=0.0)[0]
    neg = dc.score_matrices(lam, mu, rho=-0.1)[0]
    assert neg[0, 0] > base[0, 0] and neg[1, 1] > base[1, 1]  # negative rho: more 0-0 and 1-1
    assert neg[1, 0] < base[1, 0] and neg[0, 1] < base[0, 1]


def test_weights_decay_with_age_and_season():
    df, _ = _simulate(50)
    ref = pd.Timestamp("2023-06-01")
    w = dc.match_weights(df, ref, dc.DCConfig(xi=0.01))
    order = np.argsort(df["date"].to_numpy())
    assert (np.diff(w[order]) >= -1e-12).all()  # newer matches never weigh less
    old = df.assign(season="2020-21")
    w_old = dc.match_weights(old, ref, dc.DCConfig(xi=0.0, season_decay=0.8))
    assert w_old[0] == pytest.approx(0.8**2)  # 2020-21 is 2 seasons before the 2022-23 ref


def test_fit_is_deterministic():
    df, _ = _simulate(500)
    f1 = dc.fit_dc(df, pd.Timestamp("2023-06-01"), dc.DCConfig())
    f2 = dc.fit_dc(df, pd.Timestamp("2023-06-01"), dc.DCConfig())
    assert np.allclose(f1.attack, f2.attack)


def test_weighted_prior_mean_hand_sequence():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = momentum._weighted_prior_mean(vals)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)
    # weights oldest->newest 1..5 over [1..5]: (1+4+9+16+25)/15
    assert out[5] == pytest.approx(55 / 15)
    # only the last five prior values matter at index 5; the 6th value (6.0) is unseen
    assert out[5] == pytest.approx(momentum._weighted_prior_mean(np.append(vals[:5], 99.0))[5])


@pytest.mark.skipif(not build_dataset.MATCHES_PATH.exists(), reason="run build first")
def test_momentum_has_no_future_leakage():
    m = build_dataset.load_matches()
    m = m[m["date"] < "2018-01-01"].reset_index(drop=True)
    full = momentum.compute_momentum(m)
    cut = m[m["date"] < "2017-03-01"]
    part = momentum.compute_momentum(cut)
    assert np.allclose(full.loc[cut.index, "mom_diff"], part["mom_diff"])
    # removing the future also leaves the Dixon-Coles features untouched: perturb a future row
    changed = m.copy()
    changed.loc[changed.index[-1], "home_goals"] = 9
    alt = momentum.compute_momentum(changed)
    assert np.allclose(full.iloc[:-5]["mom_diff"], alt.iloc[:-5]["mom_diff"])


def test_parameter_draws_spread_and_center():
    from plpredictor.simulation import season_sim

    df, _ = _simulate(1500)
    cfg = dc.DCConfig(xi=0.0, ridge=0.5, w_e1=1.0)
    ref = pd.Timestamp("2023-06-01")
    fit = dc.fit_dc(df, ref, cfg)
    draws = dc.parameter_draws(df, ref, cfg, fit, k=200)
    rel = np.array([d.attack - d.attack.mean() + d.defence - d.defence.mean() for d in draws])
    assert rel.std(axis=0).min() > 0.01  # genuinely uncertain
    centred = rel.mean(axis=0)
    base = fit.attack - fit.attack.mean() + fit.defence - fit.defence.mean()
    assert np.abs(centred - base).max() < 0.1  # centred on the best fit
    wide = dc.parameter_draws(df, ref, cfg, fit, k=200, scale=2.0)
    assert np.std([d.attack[0] for d in wide]) > np.std([d.attack[0] for d in draws])
    # the uncertain simulation is a valid distribution
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    table = season_sim.current_table(df.iloc[:0].assign(home_goals=0, away_goals=0), teams)
    fx = season_sim.remaining_fixtures(df.iloc[:0], teams)
    per = []
    for d in draws[:40]:
        lam, mu = dc.expected_goals(d, fx["home_team"], fx["away_team"])
        per.append(dc.score_matrices(lam, mu, d.rho))
    res = season_sim.simulate_uncertain(table, fx, per, n_sims=2000)
    assert res.position_counts.sum(axis=1).tolist() == [2000] * len(teams)
    assert res.summary()["p_title"].sum() == pytest.approx(1.0)


def test_team_home_advantage_is_recovered_and_shrunk():
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(6)]
    extra = np.array([0.35, 0.0, 0.0, 0.0, 0.0, -0.3])  # T0 strong at home, T5 weak at home
    rows = []
    d0 = pd.Timestamp("2022-08-01")
    for k in range(6000):
        h, a = rng.choice(6, 2, replace=False)
        lam = np.exp(0.2 + 0.25 + extra[h])
        mu = np.exp(0.2)
        rows.append(("E0", "2022-23", d0 + pd.Timedelta(days=k % 300), teams[h], teams[a],
                     rng.poisson(lam), rng.poisson(mu)))  # fmt: skip
    df = pd.DataFrame(rows, columns=["div", "season", "date", "home_team", "away_team",
                                     "home_goals", "away_goals"])  # fmt: skip
    df["status"] = "played"
    df["home_red"] = df["away_red"] = 0.0
    ref = pd.Timestamp("2023-06-01")
    loose = dc.fit_dc(df, ref, dc.DCConfig(xi=0.0, ridge=0.01, team_home_ridge=1.0))
    assert loose.team_home[0] - loose.team_home[5] == pytest.approx(0.65, abs=0.12)
    tight = dc.fit_dc(df, ref, dc.DCConfig(xi=0.0, ridge=0.01, team_home_ridge=5000.0))
    assert np.abs(tight.team_home).max() < np.abs(loose.team_home).max() / 3  # ridge shrinks
    off = dc.fit_dc(df, ref, dc.DCConfig(xi=0.0, ridge=0.01))
    assert np.allclose(off.team_home, 0.0)  # disabled by default


def test_xg_blend_uses_proxy_and_zero_blend_ignores_it():
    df, _ = _simulate(800)
    df["home_xgp"] = df["home_goals"] * 0 + 1.0  # absurd proxy: everyone creates 1.0
    df["away_xgp"] = 1.0
    ref = pd.Timestamp("2023-06-01")
    plain = dc.fit_dc(df, ref, dc.DCConfig(xi=0.0))
    zero = dc.fit_dc(df.drop(columns=["home_xgp", "away_xgp"]), ref, dc.DCConfig(xi=0.0))
    assert np.allclose(plain.attack, zero.attack)  # blend 0 never touches the proxy
    full = dc.fit_dc(df, ref, dc.DCConfig(xi=0.0, xg_blend=1.0))
    assert np.abs(full.attack).max() < 0.05  # identical targets for everyone: no team differences


def test_xg_proxy_fit_and_leak_free_window():
    from plpredictor.features import xg_proxy

    rng = np.random.default_rng(0)
    n = 4000
    sot, off = rng.integers(1, 10, n), rng.integers(0, 12, n)
    goals = rng.poisson(0.3 * sot + 0.02 * off)
    m = pd.DataFrame({"status": "played", "season": "2019-20", "home_goals": goals,
                      "away_goals": goals, "home_sot": sot, "away_sot": sot,
                      "home_shots": sot + off, "away_shots": sot + off})  # fmt: skip
    coef = xg_proxy.fit_proxy(m)
    assert coef["sot"] == pytest.approx(0.3, abs=0.03)
    late = m.assign(season="2025-26", home_goals=99, away_goals=99)  # outside the fit window
    assert xg_proxy.fit_proxy(pd.concat([m, late])) == coef
    out = xg_proxy.add_proxy(m.head(3), coef)
    assert out["home_xgp"].notna().all()


def test_predicted_pick_score_agrees_with_winner():
    m = dc.score_matrices(np.array([2.4]), np.array([0.7]), rho=-0.05)[0]
    pick = dc.predicted_pick(m)
    assert pick["outcome"] == "H" and pick["home_goals"] > pick["away_goals"]
    assert pick["confidence"] == "Strong pick"
    # evenly matched: the call may be a draw, but score must still match the call
    even = dc.score_matrices(np.array([1.2]), np.array([1.2]), rho=-0.1)[0]
    p2 = dc.predicted_pick(even)
    if p2["outcome"] == "D":
        assert p2["home_goals"] == p2["away_goals"]
    assert 0 < p2["score_prob"] < p2["outcome_prob"] <= 1
