"""Walk-forward backtesting: never train on the future.

Test matches are processed in chunks (``refit_days`` apart). Each chunk is predicted by a
model fitted only on matches strictly before the chunk's first kickoff date, which mimics
the live app refreshing after each round of matches.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from plpredictor.evaluation import metrics

# Seasons used to tune hyper-parameters (never to report final numbers).
VALIDATION_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
TEST_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def chunk_by_days(test: pd.DataFrame, refit_days: int) -> list[pd.DataFrame]:
    """Split test matches into consecutive windows of ``refit_days`` days."""
    test = test.sort_values("date")
    start = test["date"].min()
    bucket = ((test["date"] - start).dt.days // refit_days).to_numpy()
    return [g for _, g in test.groupby(bucket)]


def walk_forward(
    fit_predict: FitPredict,
    matches: pd.DataFrame,
    seasons: list[str],
    refit_days: int = 7,
) -> pd.DataFrame:
    """Return a frame (indexed like ``matches``) with columns p_h, p_d, p_a for test matches.

    ``fit_predict(train, test)`` gets all *played* matches before the chunk (every division)
    and the chunk's matches (Premier League only) and returns an (n, 3) probability array.
    """
    played = matches[matches["status"] == "played"]
    out = []
    for season in seasons:
        test = played[(played["season"] == season) & (played["div"] == "E0")]
        if test.empty:
            continue
        for chunk in chunk_by_days(test, refit_days):
            train = played[played["date"] < chunk["date"].min()]
            p = fit_predict(train, chunk)
            out.append(pd.DataFrame(p, index=chunk.index, columns=["p_h", "p_d", "p_a"]))
    return pd.concat(out).sort_index()


def score(preds: pd.DataFrame, matches: pd.DataFrame) -> dict[str, float]:
    """Metrics for a prediction frame (rows with NaN probabilities are ignored)."""
    m = matches.loc[preds.index]
    ok = preds.notna().all(axis=1)
    return metrics.summarise(
        preds.loc[ok].to_numpy(), metrics.outcome_index(m.loc[ok, "result"])
    )
