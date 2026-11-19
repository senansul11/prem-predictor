"""Convert decimal odds into margin-free probabilities.

Bookmakers add a margin ("overround"), so 1/odds for all outcomes sums to more than 1.
We remove it by proportional normalisation (basic) or the power method (alternative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def implied_probs(h: pd.Series, d: pd.Series, a: pd.Series) -> pd.DataFrame:
    """Proportionally normalised implied probabilities. Invalid odds (<=1 or NaN) -> NaN."""
    raw = pd.concat([1.0 / h, 1.0 / d, 1.0 / a], axis=1)
    raw.columns = ["h", "d", "a"]
    valid = (pd.concat([h, d, a], axis=1) > 1.0).all(axis=1)
    raw = raw.where(valid)
    return raw.div(raw.sum(axis=1), axis=0)


def power_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Power-method margin removal: find k with sum((1/o)**k) = 1 (stretch method)."""
    inv = np.array([1 / h, 1 / d, 1 / a])
    if not np.isfinite(inv).all():
        return (np.nan, np.nan, np.nan)
    k = brentq(lambda k: (inv**k).sum() - 1.0, 0.5, 5.0)
    p = inv**k
    return tuple(p)  # type: ignore[return-value]


def two_way_prob(over: pd.Series, under: pd.Series) -> pd.Series:
    """Margin-free probability of the first outcome in a two-way market (e.g. over 2.5)."""
    valid = (over > 1.0) & (under > 1.0)
    p = (1 / over) / (1 / over + 1 / under)
    return p.where(valid)
