"""Blend Dixon-Coles and XGBoost probabilities."""

from __future__ import annotations

import numpy as np


def blend(p_dc: np.ndarray, p_xgb: np.ndarray, w_xgb: float) -> np.ndarray:
    """Linear blend: (1-w) * Dixon-Coles + w * XGBoost. Rows still sum to 1."""
    return (1 - w_xgb) * p_dc + w_xgb * p_xgb
