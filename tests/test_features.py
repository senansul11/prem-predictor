import numpy as np
import pandas as pd
import pytest

from plpredictor.data import build_dataset
from plpredictor.features import build_features as bf
from plpredictor.models.dixon_coles import DCConfig
from plpredictor.models.elo import EloParams
from plpredictor.models.xgb_model import season_weights

pytestmark = pytest.mark.skipif(not build_dataset.MATCHES_PATH.exists(), reason="run build first")


def _subset():
    m = build_dataset.load_matches()
    return m[m["date"] < "2017-06-01"].reset_index(drop=True)


def _features(m):
    empty = pd.DataFrame(np.nan, index=m.index, columns=bf.DC_FEATS)  # DC leakage tested elsewhere
    return bf.build_features(m, EloParams(), DCConfig(), dc_features=empty)


def test_features_unchanged_by_future_rows():
    m = _subset()
    full = _features(m)
    cut_date = pd.Timestamp("2016-12-01")
    part_m = m[m["date"] < cut_date]
    part = _features(part_m)
    cols = bf.form_features() + bf.ELO_FEATS + bf.MOM_FEATS
    pd.testing.assert_frame_equal(
        full.loc[part_m.index, cols], part[cols], check_exact=False, atol=1e-9
    )


def test_features_ignore_a_matchs_own_result():
    m = _subset()
    base = _features(m)
    i = len(m) // 2
    changed = m.copy()
    changed.loc[i, ["home_goals", "away_goals", "home_sot", "away_sot"]] = [7, 0, 20, 0]
    alt = _features(changed)
    cols = bf.form_features() + bf.ELO_FEATS + bf.MOM_FEATS
    # the match's own features are identical; only later rows may differ
    pd.testing.assert_series_equal(base.loc[i, cols], alt.loc[i, cols], check_exact=False)
    earlier = base.loc[: i - 1, cols]
    pd.testing.assert_frame_equal(earlier, alt.loc[: i - 1, cols], check_exact=False)


def test_season_weights_decrease_with_age():
    df = pd.DataFrame({"season": ["2023-24", "2024-25", "2025-26"]})
    w = season_weights(df, pd.Timestamp("2025-12-01"), 0.8)
    assert w[0] < w[1] < w[2] == 1.0
    assert (season_weights(df, pd.Timestamp("2025-12-01"), 1.0) == 1.0).all()
