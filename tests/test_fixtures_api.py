import numpy as np
import pandas as pd
import pytest

from plpredictor.data.sources import fixtures_api as fa
from plpredictor.data.team_names import UnmappedTeamError


def _m(home, away, date, odds=(1.5, 4.0, 6.0), mid=1):
    return {"match_id": mid, "match_date": date, "game_week": 12, "status_localized": "Scheduled",
            "home_team": {"team_name": home}, "away_team": {"team_name": away},
            "odds": {"home_win": odds[0], "draw": odds[1], "away_win": odds[2]}}  # fmt: skip


def test_both_response_shapes_and_name_mapping():
    ms = [_m("Manchester City", "Fulham", "2026-11-21 12:30:00")]
    assert fa.extract_matches({"data": ms}) == fa.extract_matches({"data": {"matches": ms}})
    df = fa.parse_fixtures(ms)
    assert df.loc[0, "home_team"] == "Manchester City" and df.loc[0, "away_team"] == "Fulham"


def test_zero_odds_become_nan_and_unmapped_team_raises():
    df = fa.parse_fixtures([_m("Arsenal", "Chelsea", "2026-11-21 12:30:00", odds=(0, 0, 0))])
    assert df[["api_odds_h", "api_odds_d", "api_odds_a"]].isna().all().all()
    with pytest.raises(UnmappedTeamError):
        fa.parse_fixtures([_m("Nowhere FC", "Chelsea", "2026-11-21 12:30:00")])


def test_full_remaining_keeps_undated_pairs():
    teams = ["A", "B", "C"]
    played = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    dated = pd.DataFrame({"home_team": ["B"], "away_team": ["A"],
                          "date": [pd.Timestamp("2026-12-01")], "game_week": [14]})  # fmt: skip
    out = fa.full_remaining(played, teams, dated)
    assert len(out) == 5  # 6 pairs minus 1 played
    assert out["dated"].sum() == 1
    assert out.loc[out["dated"], "date"].notna().all()
    assert out.loc[~out["dated"], "date"].isna().all()
    none = fa.full_remaining(played, teams, None)
    assert len(none) == 5 and not none["dated"].any() and np.isnan(none["game_week"]).all()
