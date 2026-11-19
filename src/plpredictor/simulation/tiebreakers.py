"""League ordering rules.

Premier League: points, then goal difference, then goals scored. Further steps (head-to-head
points and away goals, then a play-off) are not modelled, so any team still level is ordered
randomly, which is a fair proxy.
"""

from __future__ import annotations

import numpy as np


def rank_positions(
    pts: np.ndarray, gd: np.ndarray, gf: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return ``order`` of shape (sims, teams): team indices from 1st place to last."""
    key = pts * 1e8 + (gd + 500) * 1e4 + gf + rng.random(pts.shape) * 0.5
    return np.argsort(-key, axis=1)
