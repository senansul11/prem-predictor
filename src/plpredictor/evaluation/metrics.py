"""Probabilistic forecast metrics for home/draw/away outcomes.

Probability arrays have shape (n, 3) in the order [home, draw, away]. Outcomes are
integer indices 0=home win, 1=draw, 2=away win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def outcome_index(result: pd.Series) -> np.ndarray:
    """Map 'H'/'D'/'A' to 0/1/2."""
    return result.map(OUTCOME_INDEX).to_numpy(dtype=int)


def _onehot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), 3))
    out[np.arange(len(y)), y] = 1.0
    return out


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score (lower is better). Respects the H < D < A ordering.

    RPS = 1/(r-1) * sum_k (cumulative forecast_k - cumulative outcome_k)^2, with r=3 outcomes.
    """
    cp = np.cumsum(p, axis=1)[:, :2]
    co = np.cumsum(_onehot(y), axis=1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2.0))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log probability given to the actual outcome."""
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error against the one-hot outcome."""
    return float(np.mean(np.sum((p - _onehot(y)) ** 2, axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def summarise(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """All headline metrics in one dict."""
    return {
        "n": int(len(y)),
        "rps": rps(p, y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
    }


def calibration_bins(p: np.ndarray, y: np.ndarray, outcome: int = 0, bins: int = 10):
    """Reliability table for one outcome: mean predicted vs observed frequency per bin."""
    pred = p[:, outcome]
    obs = (y == outcome).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "mean_pred": pred[m].mean(),
                    "observed": obs[m].mean(),
                    "n": int(m.sum()),
                }
            )
    return pd.DataFrame(rows)
