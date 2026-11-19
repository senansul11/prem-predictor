"""Validation gates and change detection for the daily refresh (no network)."""

import pandas as pd
import pytest

from plpredictor import refresh
from plpredictor.data import schema
from plpredictor.data.validate import check_not_truncated, validate_matches

CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "E0,08/08/2015,Arsenal,Chelsea,1,0,H\n"
    "E0,15/08/2015,Everton,Fulham,2,2,D\n"
    "E0,22/08/2015,Liverpool,Man City,0,1,A\n"
)


def _frame(csv: str = CSV) -> pd.DataFrame:
    return schema.clean_season(schema.read_raw_csv(csv.encode()), 2015, "E0")


def test_hash_change_detection():
    a = {"E0:2026": refresh.sha256(b"one"), "fixtures": refresh.sha256(b"x")}
    assert refresh.changed(a, a) == []  # nothing new: refresh would stop early
    b = {**a, "E0:2026": refresh.sha256(b"two")}
    assert refresh.changed(a, b) == ["E0:2026"]
    assert refresh.changed({}, a) == ["E0:2026", "fixtures"]  # first ever run


def test_truncated_download_is_blocked():
    full = _frame()
    cut = _frame("\n".join(CSV.splitlines()[:2]) + "\n")  # last two matches missing
    assert check_not_truncated(None, cut) == []  # first run: nothing to compare with
    assert check_not_truncated(full, full) == []  # same data is fine
    assert check_not_truncated(cut, full) == []  # more matches is fine
    problems = check_not_truncated(full, cut)
    assert problems and "truncated" in problems[0]


def test_corrupted_rows_are_blocked_by_validation():
    ok = _frame()
    assert validate_matches(ok, expect_teams=None) == []
    bad_score = ok.copy()
    bad_score.loc[0, "away_goals"] = 5  # away "wins" but the result still says H
    assert any("score" in p for p in validate_matches(bad_score, expect_teams=None))
    dup = pd.concat([ok, ok.iloc[[0]]])
    assert any("duplicate" in p for p in validate_matches(dup, expect_teams=None))
    negative = ok.copy()
    negative.loc[1, "away_goals"] = -1
    assert any("negative" in p for p in validate_matches(negative, expect_teams=None))


def test_unknown_team_in_download_fails_loudly():
    from plpredictor.data.team_names import UnmappedTeamError

    with pytest.raises(UnmappedTeamError):
        _frame(CSV.replace("Fulham", "Brand New FC"))


def test_played_count_must_not_drop_between_runs():
    assert refresh.gate_not_regressing({"matches_played": 100}, 110) == []
    assert refresh.gate_not_regressing({"matches_played": 100}, 100) == []
    assert refresh.gate_not_regressing({"matches_played": 100}, 60)  # truncated
    assert refresh.gate_not_regressing(None, 10) == []


def test_result_column_contradicting_the_score_is_blocked():
    bad = CSV.replace("Arsenal,Chelsea,1,0,H", "Arsenal,Chelsea,1,0,A")  # 1-0 but FTR says away win
    with pytest.raises(ValueError, match="FTR contradicts"):
        _frame(bad)
    _frame()  # the untouched file is accepted
