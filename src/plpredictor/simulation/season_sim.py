"""Vectorised Monte Carlo season simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.simulation.tiebreakers import rank_positions


def current_table(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Points / GD / goals from the played matches of one season (all teams listed)."""
    t = pd.DataFrame(0, index=teams, columns=["played", "won", "drawn", "lost", "gf", "ga"])
    for r in played.itertuples():
        for team, f, a in (
            (r.home_team, r.home_goals, r.away_goals),
            (r.away_team, r.away_goals, r.home_goals),
        ):
            t.loc[team, "played"] += 1
            t.loc[team, "gf"] += int(f)
            t.loc[team, "ga"] += int(a)
            t.loc[team, "won" if f > a else "drawn" if f == a else "lost"] += 1
    t["gd"] = t["gf"] - t["ga"]
    t["pts"] = 3 * t["won"] + t["drawn"]
    return t.sort_values(["pts", "gd", "gf"], ascending=False)


def remaining_fixtures(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Double round-robin pairs not yet played (no dates: used when no schedule feed)."""
    done = set(zip(played["home_team"], played["away_team"], strict=True))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    return pd.DataFrame(rows, columns=["home_team", "away_team"])


@dataclass
class SimResult:
    teams: list[str]
    position_counts: np.ndarray  # (teams, positions): number of sims
    pts_mean: np.ndarray
    gd_mean: np.ndarray
    n_sims: int

    def summary(self, europe_places: int = 5, relegation_places: int = 3) -> pd.DataFrame:
        p = self.position_counts / self.n_sims
        n = len(self.teams)
        pos = np.arange(1, n + 1)
        out = pd.DataFrame(
            {
                "team": self.teams,
                "p_title": p[:, 0],
                "p_top4": p[:, :4].sum(axis=1),
                "p_top5": p[:, :europe_places].sum(axis=1),
                "p_relegation": p[:, n - relegation_places :].sum(axis=1),
                "exp_points": self.pts_mean,
                "exp_gd": self.gd_mean,
                "mean_position": (p * pos).sum(axis=1),
            }
        )
        return out.sort_values("exp_points", ascending=False).reset_index(drop=True)

    def position_distribution(self) -> pd.DataFrame:
        cols = [f"pos_{i}" for i in range(1, len(self.teams) + 1)]
        return pd.DataFrame(self.position_counts / self.n_sims, index=self.teams, columns=cols)


def simulate(
    table: pd.DataFrame,
    fixtures: pd.DataFrame,
    matrices: np.ndarray,
    n_sims: int = config.N_SIMS,
    seed: int = config.SEED,
    locked: dict[int, tuple[int, int]] | None = None,
    batch: int = 2500,
) -> SimResult:
    """Sample every remaining fixture ``n_sims`` times and rank the resulting tables.

    ``matrices`` has shape (n_fixtures, G+1, G+1) (home goals x away goals). ``locked`` maps a
    fixture's row position to a fixed (home, away) score for what-if scenarios.
    """
    rng = np.random.default_rng(seed)
    teams = list(table.index)
    t_idx = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    hi = fixtures["home_team"].map(t_idx).to_numpy()
    ai = fixtures["away_team"].map(t_idx).to_numpy()
    nf, g1 = matrices.shape[0], matrices.shape[1]
    cdf = np.cumsum(matrices.reshape(nf, -1), axis=1)
    cdf /= cdf[:, -1:]
    H = np.zeros((nf, nt))
    A = np.zeros((nf, nt))
    H[np.arange(nf), hi] = 1
    A[np.arange(nf), ai] = 1

    base = table[["pts", "gd", "gf"]].to_numpy(dtype=float)
    counts = np.zeros((nt, nt))
    pts_sum = np.zeros(nt)
    gd_sum = np.zeros(nt)
    done = 0
    while done < n_sims:
        s = min(batch, n_sims - done)
        u = rng.random((s, nf))
        idx = np.empty((s, nf), dtype=int)
        for f in range(nf):
            idx[:, f] = np.minimum(np.searchsorted(cdf[f], u[:, f]), cdf.shape[1] - 1)
        hg, ag = idx // g1, idx % g1
        if locked:
            for f, (lh, la) in locked.items():
                hg[:, f], ag[:, f] = lh, la
        hp = 3 * (hg > ag) + (hg == ag)
        ap = 3 * (ag > hg) + (hg == ag)
        pts = base[:, 0] + hp @ H + ap @ A
        gf = base[:, 2] + hg @ H + ag @ A
        gd = base[:, 1] + (hg - ag) @ H + (ag - hg) @ A
        order = rank_positions(pts, gd, gf, rng)
        for pos in range(nt):
            counts[:, pos] += np.bincount(order[:, pos], minlength=nt)
        pts_sum += pts.sum(axis=0)
        gd_sum += gd.sum(axis=0)
        done += s
    return SimResult(teams, counts, pts_sum / n_sims, gd_sum / n_sims, n_sims)


def simulate_uncertain(
    table: pd.DataFrame,
    fixtures: pd.DataFrame,
    matrices_per_draw: list[np.ndarray],
    n_sims: int = config.N_SIMS,
    seed: int = config.SEED,
) -> SimResult:
    """Parameter-uncertainty version: each draw of team strengths gets ``n_sims/k`` seasons.

    Mixing many plausible rating sets widens the spread of final tables, so title and
    relegation odds are no longer overconfident.
    """
    k = len(matrices_per_draw)
    per = max(n_sims // k, 1)
    counts = None
    pts = gd = 0.0
    total = 0
    for i, mats in enumerate(matrices_per_draw):
        r = simulate(table, fixtures, mats, n_sims=per, seed=seed + i, batch=per)
        counts = r.position_counts if counts is None else counts + r.position_counts
        pts = pts + r.pts_mean * per
        gd = gd + r.gd_mean * per
        total += per
    return SimResult(list(table.index), counts, pts / total, gd / total, total)
