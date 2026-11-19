import numpy as np
import pandas as pd
import pytest

from plpredictor import config
from plpredictor.data import build_dataset, odds, schema
from plpredictor.data.team_names import UnmappedTeamError, canonical
from plpredictor.data.validate import validate_matches

OLD = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HR,AR,PSH,PSD,PSA,PSCH,PSCD,PSCA,"
    "BbAvH,BbAvD,BbAvA,BbAHh,BbAv>2.5,BbAv<2.5\n"
    "E0,08/08/2015,Man United,Tottenham,1,0,H,10,8,4,3,0,0,1.8,3.6,4.8,1.75,3.7,5.0,"
    "1.8,3.5,4.6,-0.75,1.9,1.9\n"
    "E0,17/05/16,Leicester,Everton,3,1,H,12,6,6,3,0,1,1.9,3.5,4.4,1.8,3.6,4.5,"
    "1.9,3.5,4.4,-0.5,1.7,2.1\n"
    ",,,,,,,,,,,,,,,,,,,,,,,,\n"
)
NEW = (
    "Div,Date,Time,HomeTeam,AwayTeam,HG,AG,Res,HxG,AxG,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA,AHh,"
    "Avg>2.5,Avg<2.5\n"
    "E0,21/08/2026,20:00,Arsenal,Coventry,3,0,H,1.9,0.2,1.2,7,13,1.22,7.4,18,-2,1.55,2.4\n"
)


def test_old_format_and_two_year_forms():
    df = schema.clean_season(schema.read_raw_csv(OLD.encode()), 2015, "E0")
    assert len(df) == 2  # blank row dropped
    assert df["date"].tolist() == [pd.Timestamp("2015-08-08"), pd.Timestamp("2016-05-17")]
    assert df["kickoff"].isna().all()
    assert df.loc[0, "home_team"] == "Manchester United"
    assert df.loc[0, "mkt_src"] == "pinnacle"
    assert df.loc[0, "ah_line"] == -0.75  # BbAHh alias
    assert df.loc[0, "match_id"] == "2015-16_2015-08-08_Manchester United_Tottenham Hotspur"


def test_new_format_aliases_and_missing_pinnacle():
    df = schema.clean_season(schema.read_raw_csv(NEW), 2026, "E0")
    r = df.iloc[0]
    assert (r.home_goals, r.away_goals, r.result) == (3, 0, "H")  # HG/AG/Res aliases
    assert r.mkt_src == "average" and r.close_src == "average"
    assert r.home_xg == 1.9 and r.kickoff == "20:00"


def test_missing_stays_nan_not_zero():
    df = schema.clean_season(schema.read_raw_csv(NEW), 2026, "E0")
    assert np.isnan(df.loc[0, "home_shots"])


def test_latin1_fallback():
    raw = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,Referee\n"
        "E0,08/08/2015,Arsenal,Chelsea,1,1,Andr\xe9\n"
    )
    df = schema.read_raw_csv(raw.encode("latin-1"))
    assert df.loc[0, "Referee"] == "Andr\xe9"


def test_covid_season_window_allowed_but_bad_date_fails():
    ok = pd.Series([pd.Timestamp("2020-07-26"), pd.Timestamp("2019-08-09")])
    schema.check_season_window(ok, 2019)
    with pytest.raises(ValueError):
        schema.check_season_window(pd.Series([pd.Timestamp("2021-01-01")]), 2019)


def test_unmapped_team_raises():
    with pytest.raises(UnmappedTeamError):
        canonical("Nowhere FC")
    bad = OLD.replace("Everton", "Nowhere FC")
    with pytest.raises(UnmappedTeamError):
        schema.clean_season(schema.read_raw_csv(bad.encode()), 2015, "E0")


def test_odds_sum_to_one_and_nan_propagates():
    p = odds.implied_probs(pd.Series([2.0, np.nan]), pd.Series([3.5, 3.0]), pd.Series([4.0, 3.0]))
    assert p.iloc[0].sum() == pytest.approx(1.0)
    assert p.iloc[1].isna().all()
    ph, pd_, pa = odds.power_probs(2.0, 3.5, 4.0)
    assert ph + pd_ + pa == pytest.approx(1.0)


def test_validate_catches_problems():
    df = schema.clean_season(schema.read_raw_csv(OLD.encode()), 2015, "E0")
    assert validate_matches(df, expect_teams=None) == []
    bad = pd.concat([df, df.iloc[[0]]])
    assert any("duplicate" in p for p in validate_matches(bad, expect_teams=None))
    wrong = df.copy()
    wrong.loc[0, "result"] = "A"
    assert any("score" in p for p in validate_matches(wrong, expect_teams=None))


def test_upsert_is_idempotent():
    df = schema.clean_season(schema.read_raw_csv(OLD.encode()), 2015, "E0")
    once = build_dataset.upsert(None, df)
    twice = build_dataset.upsert(once, df)
    assert len(once) == len(twice) == 2
    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.skipif(not build_dataset.MATCHES_PATH.exists(), reason="run build first")
def test_real_data_spot_checks():
    m = build_dataset.load_matches(config.LEAGUE)
    assert (m["status"] == "played").sum() == 4280
    s = m[m["season"] == "2015-16"]
    home = s.assign(pts=s["result"].map({"H": 3, "D": 1, "A": 0}), team=s["home_team"])
    away = s.assign(pts=s["result"].map({"A": 3, "D": 1, "H": 0}), team=s["away_team"])
    table = pd.concat([home, away]).groupby("team")["pts"].sum()
    assert table["Leicester City"] == 81  # 2015-16 champions
    assert table["Aston Villa"] == 17
    assert len(table) == 20
