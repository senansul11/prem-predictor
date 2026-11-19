"""Dixon-Coles goals model.

Idea (plain words): every team has an *attack* strength and a *defence* strength. The goals a
team scores are Poisson-distributed (a "count of rare events" distribution) with an average
that depends on its attack, the opponent's defence and a home-advantage boost::

    home_goals ~ Poisson(exp(mu0 + home_adv + attack_home - defence_away))
    away_goals ~ Poisson(exp(mu0 +            attack_away - defence_home))

Plain Poisson gets 0-0, 1-0, 0-1 and 1-1 slightly wrong, so Dixon & Coles multiply those four
scorelines by a correction factor governed by ``rho``. Recent matches count more through an
exponential time decay (``xi``) and an optional extra per-season weight. Championship matches
are included (with a league-level goal offset) so promoted teams arrive with a rating.
A small ridge penalty shrinks every team towards "average" which stabilises thin data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from plpredictor import config


@dataclass
class DCConfig:
    xi: float = 0.001  # time decay per day (half-life = ln2/xi days)
    season_decay: float = 1.0  # extra weight multiplier per season back (1 = off)
    ridge: float = 3.0  # shrinkage of attack/defence towards 0
    w_e1: float = 0.5  # weight of Championship matches
    red_w: float = 1.0  # weight multiplier for matches containing a red card
    use_momentum: bool = False  # lambda *= exp(+-beta * momentum_diff)
    xg_blend: float = 0.0  # share of the shot-based xG proxy in the scoring target (0 = goals)
    team_home_ridge: float = 0.0  # >0 enables per-team home advantage (shrunk by this ridge)
    max_goals: int = config.MATRIX_MAX_GOALS


@dataclass
class DCFit:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    mu0: float
    rho: float
    c_e1: float
    beta: float
    team_home: np.ndarray | None = None  # per-team extra home advantage
    cfg: DCConfig = field(default_factory=DCConfig)

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}


def match_weights(df: pd.DataFrame, ref_date: pd.Timestamp, cfg: DCConfig) -> np.ndarray:
    """Weight of each training match: time decay x season weight x div weight x red-card."""
    age = (ref_date - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    w = np.exp(-cfg.xi * age)
    ref_start = config.season_start_year(ref_date.date())
    start = df["season"].str[:4].astype(int).to_numpy()
    w = w * cfg.season_decay ** np.clip(ref_start - start, 0, None)
    w = np.where(df["div"].to_numpy() == config.LEAGUE, w, w * cfg.w_e1)
    if cfg.red_w != 1.0:
        red = (df["home_red"].fillna(0) + df["away_red"].fillna(0)).to_numpy() > 0
        w = np.where(red, w * cfg.red_w, w)
    return w


def _tau_terms(x, y, lam, mu, rho):
    """Dixon-Coles correction: log tau and its derivatives wrt (eta_h, eta_a, rho)."""
    n = len(x)
    log_tau = np.zeros(n)
    d_h, d_a, d_r = np.zeros(n), np.zeros(n), np.zeros(n)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t = np.clip(1 - lam[m00] * mu[m00] * rho, 1e-9, None)
    log_tau[m00] = np.log(t)
    d_h[m00] = d_a[m00] = -lam[m00] * mu[m00] * rho / t
    d_r[m00] = -lam[m00] * mu[m00] / t
    t = np.clip(1 + lam[m01] * rho, 1e-9, None)
    log_tau[m01] = np.log(t)
    d_h[m01] = lam[m01] * rho / t
    d_r[m01] = lam[m01] / t
    t = np.clip(1 + mu[m10] * rho, 1e-9, None)
    log_tau[m10] = np.log(t)
    d_a[m10] = mu[m10] * rho / t
    d_r[m10] = mu[m10] / t
    t = np.clip(1 - rho, 1e-9, None)
    log_tau[m11] = np.log(t)
    d_r[m11] = -1.0 / t
    return log_tau, d_h, d_a, d_r


def _problem(
    train: pd.DataFrame,
    ref_date: pd.Timestamp,
    cfg: DCConfig,
    init: DCFit | None = None,
):
    """Build the objective (loss + gradient), start point and bounds for one training set."""
    train = train[train["status"] == "played"]
    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = train["home_team"].map(idx).to_numpy()
    ai = train["away_team"].map(idx).to_numpy()
    x = train["home_goals"].to_numpy(dtype=float)
    y = train["away_goals"].to_numpy(dtype=float)
    w = match_weights(train, ref_date, cfg)
    e1 = (train["div"].to_numpy() != config.LEAGUE).astype(float)
    md = train["mom_diff"].to_numpy(dtype=float) if cfg.use_momentum else np.zeros(len(train))
    x_t, y_t = x, y  # Poisson scoring targets (goals, optionally blended with the xG proxy)
    if cfg.xg_blend > 0:
        xp = train["home_xgp"].to_numpy(dtype=float)
        yp = train["away_xgp"].to_numpy(dtype=float)
        x_t = np.where(np.isnan(xp), x, (1 - cfg.xg_blend) * x + cfg.xg_blend * xp)
        y_t = np.where(np.isnan(yp), y, (1 - cfg.xg_blend) * y + cfg.xg_blend * yp)
    use_hv = cfg.team_home_ridge > 0

    theta0 = np.zeros(3 * n + 5)
    theta0[2 * n] = 0.25  # home advantage
    theta0[2 * n + 1] = 0.25  # mu0
    if init is not None:
        old = init.index
        for t, i in idx.items():
            if t in old:
                theta0[i], theta0[n + i] = init.attack[old[t]], init.defence[old[t]]
        theta0[2 * n : 2 * n + 5] = [init.home_adv, init.mu0, init.rho, init.c_e1, init.beta]
        if init.team_home is not None and use_hv:
            for t, i in idx.items():
                if t in old:
                    theta0[2 * n + 5 + i] = init.team_home[old[t]]

    def fun(th):
        a, d = th[:n], th[n : 2 * n]
        h, mu0, rho, c1, beta = th[2 * n : 2 * n + 5]
        hv = th[2 * n + 5 :]
        eta_h = mu0 + h + hv[hi] + a[hi] - d[ai] + c1 * e1 + beta * md
        eta_a = mu0 + a[ai] - d[hi] + c1 * e1 - beta * md
        lam, mu = np.exp(eta_h), np.exp(eta_a)
        log_tau, th_h, th_a, th_r = _tau_terms(x, y, lam, mu, rho)
        ll = x_t * eta_h - lam + y_t * eta_a - mu + log_tau  # dropped constant log-factorials
        loss = -np.sum(w * ll) + cfg.ridge * (np.sum(a**2) + np.sum(d**2))
        loss += cfg.team_home_ridge * np.sum(hv**2)
        g_h = w * (x_t - lam + th_h)  # d ll / d eta_h
        g_a = w * (y_t - mu + th_a)
        grad = np.zeros_like(th)
        grad[:n] = -(np.bincount(hi, g_h, n) + np.bincount(ai, g_a, n)) + 2 * cfg.ridge * a
        grad[n : 2 * n] = (np.bincount(ai, g_h, n) + np.bincount(hi, g_a, n)) + 2 * cfg.ridge * d
        grad[2 * n] = -np.sum(g_h)
        grad[2 * n + 1] = -np.sum(g_h + g_a)
        grad[2 * n + 2] = -np.sum(w * th_r)
        grad[2 * n + 3] = -np.sum((g_h + g_a) * e1)
        grad[2 * n + 4] = -np.sum((g_h - g_a) * md)
        grad[2 * n + 5 :] = -np.bincount(hi, g_h, n) + 2 * cfg.team_home_ridge * hv
        return loss, grad

    bounds = [(None, None)] * (2 * n) + [
        (None, None),  # home advantage
        (None, None),  # mu0
        (-0.25, 0.25),  # rho
        (None, None) if e1.any() else (0.0, 0.0),  # league offset
        (None, None) if cfg.use_momentum else (0.0, 0.0),  # momentum beta
    ] + [(None, None) if use_hv else (0.0, 0.0)] * n  # per-team home advantage
    return fun, theta0, bounds, teams, n


def _to_fit(th: np.ndarray, teams: list[str], n: int, cfg: DCConfig) -> DCFit:
    return DCFit(
        teams=teams, attack=th[:n].copy(), defence=th[n : 2 * n].copy(), home_adv=th[2 * n],
        mu0=th[2 * n + 1], rho=th[2 * n + 2], c_e1=th[2 * n + 3], beta=th[2 * n + 4],
        team_home=th[2 * n + 5 : 3 * n + 5].copy(), cfg=cfg,
    )  # fmt: skip


def fit_dc(
    train: pd.DataFrame,
    ref_date: pd.Timestamp,
    cfg: DCConfig,
    init: DCFit | None = None,
) -> DCFit:
    """Fit by weighted maximum likelihood (L-BFGS-B with an analytic gradient)."""
    fun, theta0, bounds, teams, n = _problem(train, ref_date, cfg, init)
    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})  # fmt: skip
    return _to_fit(res.x, teams, n, cfg)


def parameter_draws(
    train: pd.DataFrame,
    ref_date: pd.Timestamp,
    cfg: DCConfig,
    fit: DCFit,
    k: int = 200,
    seed: int = config.SEED,
    scale: float = 1.0,
) -> list[DCFit]:
    """Sample ``k`` plausible parameter sets around the best fit (Laplace approximation).

    The curvature (Hessian) of the loss at the optimum says how tightly the data pins each
    parameter down; its inverse is the covariance of a normal distribution we draw from.
    Teams with little data get wide spreads. ``scale`` > 1 widens the spread.
    """
    fun, _, bounds, teams, n = _problem(train, ref_date, cfg, fit)
    idx = fit.index
    th = np.zeros(3 * n + 5)
    for t, i in ((t, i) for i, t in enumerate(teams)):
        th[i], th[n + i] = fit.attack[idx[t]], fit.defence[idx[t]]
        if fit.team_home is not None:
            th[2 * n + 5 + i] = fit.team_home[idx[t]]
    th[2 * n : 2 * n + 5] = [fit.home_adv, fit.mu0, fit.rho, fit.c_e1, fit.beta]
    free = np.array([not (b[0] is not None and b[0] == b[1]) for b in bounds])
    fi = np.flatnonzero(free)
    eps = 1e-4
    H = np.zeros((len(fi), len(fi)))
    for c, j in enumerate(fi):
        up, dn = th.copy(), th.copy()
        up[j] += eps
        dn[j] -= eps
        H[:, c] = (fun(up)[1][fi] - fun(dn)[1][fi]) / (2 * eps)
    H = (H + H.T) / 2
    vals, vecs = np.linalg.eigh(H)
    vals = np.clip(vals, 1e-3, None)
    cov = (vecs / vals) @ vecs.T * scale**2
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros(len(fi)), cov, size=k, method="eigh")
    out = []
    for d in draws:
        t = th.copy()
        t[fi] += d
        t[2 * n + 2] = np.clip(t[2 * n + 2], -0.25, 0.25)
        out.append(_to_fit(t, teams, n, cfg))
    return out


def expected_goals(fit: DCFit, home: pd.Series, away: pd.Series, mom_diff=None):
    """Return (lambda_home, lambda_away). Unknown teams are treated as average (0, 0)."""
    idx = fit.index
    a = np.array([fit.attack[idx[t]] if t in idx else 0.0 for t in home])
    d_h = np.array([fit.defence[idx[t]] if t in idx else 0.0 for t in home])
    a_a = np.array([fit.attack[idx[t]] if t in idx else 0.0 for t in away])
    d_a = np.array([fit.defence[idx[t]] if t in idx else 0.0 for t in away])
    md = np.zeros(len(a)) if mom_diff is None else np.asarray(mom_diff, dtype=float)
    hv = np.zeros(len(a))
    if fit.team_home is not None:
        hv = np.array([fit.team_home[idx[t]] if t in idx else 0.0 for t in home])
    lam = np.exp(fit.mu0 + fit.home_adv + hv + a - d_a + fit.beta * md)
    mu = np.exp(fit.mu0 + a_a - d_h - fit.beta * md)
    return lam, mu


def score_matrices(lam, mu, rho: float, max_goals: int = config.MATRIX_MAX_GOALS) -> np.ndarray:
    """Scoreline probabilities, shape (n, G+1, G+1): [match, home goals, away goals]."""
    lam, mu = np.atleast_1d(lam), np.atleast_1d(mu)
    k = np.arange(max_goals + 1)
    logfact = gammaln(k + 1)
    ph = np.exp(k[None, :] * np.log(lam[:, None]) - lam[:, None] - logfact[None, :])
    pa = np.exp(k[None, :] * np.log(mu[:, None]) - mu[:, None] - logfact[None, :])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1 - lam * mu * rho
    m[:, 0, 1] *= 1 + lam * rho
    m[:, 1, 0] *= 1 + mu * rho
    m[:, 1, 1] *= 1 - rho
    return m / m.sum(axis=(1, 2), keepdims=True)  # renormalise the truncated tail


def matrix_summary(m: np.ndarray) -> pd.DataFrame:
    """W/D/L, over 2.5, BTTS and expected goals from scoreline matrices."""
    g = m.shape[1]
    i, j = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    return pd.DataFrame(
        {
            "p_h": (m * (i > j)).sum(axis=(1, 2)),
            "p_d": (m * (i == j)).sum(axis=(1, 2)),
            "p_a": (m * (i < j)).sum(axis=(1, 2)),
            "p_over25": (m * ((i + j) > 2)).sum(axis=(1, 2)),
            "p_btts": (m * ((i > 0) & (j > 0))).sum(axis=(1, 2)),
            "xg_h": (m * i).sum(axis=(1, 2)),
            "xg_a": (m * j).sum(axis=(1, 2)),
        }
    )


def top_scorelines(matrix: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    """Most likely scorelines for one match matrix."""
    flat = np.argsort(matrix, axis=None)[::-1][:n]
    g = matrix.shape[1]
    return [(int(f // g), int(f % g), float(matrix.flat[f])) for f in flat]


def predicted_pick(matrix: np.ndarray) -> dict:
    """The model's headline call for one match.

    The predicted result is the most likely of home win / draw / away win. The predicted
    score is the most likely scoreline *within that result*, so score and winner always agree.
    ``confidence`` is a plain-words label based on the winning probability.
    """
    g = matrix.shape[0]
    i, j = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    masks = {"H": i > j, "D": i == j, "A": i < j}
    probs = {k: float(matrix[m].sum()) for k, m in masks.items()}
    outcome = max(probs, key=probs.get)
    sub = np.where(masks[outcome], matrix, -1.0)
    hg, ag = np.unravel_index(np.argmax(sub), sub.shape)
    top = probs[outcome]
    label = "Strong pick" if top >= 0.55 else "Lean" if top >= 0.45 else "Toss-up"
    return {
        "outcome": outcome,
        "home_goals": int(hg),
        "away_goals": int(ag),
        "outcome_prob": top,
        "score_prob": float(matrix[hg, ag]),
        "confidence": label,
    }


class DixonColes:
    """Convenience wrapper: fit on past matches, then predict any fixtures."""

    def __init__(self, cfg: DCConfig | None = None) -> None:
        self.cfg = cfg or DCConfig()
        self.fit_: DCFit | None = None

    def fit(self, train: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> DixonColes:
        ref = ref_date if ref_date is not None else train["date"].max() + pd.Timedelta(days=1)
        self.fit_ = fit_dc(train, ref, self.cfg, init=self.fit_)
        return self

    def matrices(self, fixtures: pd.DataFrame) -> np.ndarray:
        md = fixtures["mom_diff"] if self.cfg.use_momentum and "mom_diff" in fixtures else None
        lam, mu = expected_goals(self.fit_, fixtures["home_team"], fixtures["away_team"], md)
        return score_matrices(lam, mu, self.fit_.rho, self.cfg.max_goals)

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Summary probabilities for fixtures (index preserved)."""
        out = matrix_summary(self.matrices(fixtures))
        out.index = fixtures.index
        return out

    def team_table(self) -> pd.DataFrame:
        f = self.fit_
        return pd.DataFrame({"team": f.teams, "attack": f.attack, "defence": f.defence}).assign(
            overall=lambda d: d["attack"] + d["defence"]
        ).sort_values("overall", ascending=False).reset_index(drop=True)
