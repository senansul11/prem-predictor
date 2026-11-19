"""Map the three football-data.co.uk header layouts onto one canonical schema."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

from plpredictor import config
from plpredictor.data import odds
from plpredictor.data.team_names import canonical

# Alias pairs that mean the same thing in different seasons.
ALIASES: dict[str, str] = {
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
    "PH": "PSH",
    "PD": "PSD",
    "PA": "PSA",
}

RESULT_COLS = [
    "match_id", "div", "season", "date", "kickoff", "home_team", "away_team",
    "home_goals", "away_goals", "result", "home_xg", "away_xg",
    "home_shots", "away_shots", "home_sot", "away_sot", "home_red", "away_red",
    "p_mkt_h", "p_mkt_d", "p_mkt_a", "mkt_src",
    "p_close_h", "p_close_d", "p_close_a", "close_src",
    "p_over25", "ah_line", "ah_close_line", "status", "source", "ingested_at",
]  # fmt: skip

_STAT_MAP = {
    "HxG": "home_xg", "AxG": "away_xg", "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot", "HR": "home_red", "AR": "away_red",
}  # fmt: skip


def read_raw_csv(source: bytes | str | io.BytesIO) -> pd.DataFrame:
    """Read a football-data CSV with utf-8 -> latin-1 fallback; drop empty rows/columns."""
    if isinstance(source, io.BytesIO):
        source = source.getvalue()
    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = source.decode("latin-1")
    else:
        text = source
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse day-first dates in either dd/mm/yyyy or dd/mm/yy form."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = out.isna()
    out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    if out.isna().any():
        bad = s[out.isna()].head(3).tolist()
        raise ValueError(f"Unparseable dates: {bad}")
    return out


def check_season_window(dates: pd.Series, start_year: int) -> None:
    """All dates must fall between 15 Jul of the start year and 31 Jul of the next.

    The window deliberately runs to the end of July to allow the COVID 2019-20 season
    (which finished on 26 July 2020).
    """
    lo, hi = pd.Timestamp(start_year, 7, 15), pd.Timestamp(start_year + 1, 7, 31)
    out = dates[(dates < lo) | (dates > hi)]
    if len(out):
        raise ValueError(f"{len(out)} dates outside season window {lo.date()}..{hi.date()}")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """First non-null value across alias columns, row by row."""
    out = _num(df, cols[0])
    for c in cols[1:]:
        out = out.combine_first(_num(df, c))
    return out


def _market(df: pd.DataFrame, prefixes: list[tuple[str, list[str]]], tag: str):
    """Build margin-free H/D/A probs from the first available bookmaker group, per row.

    ``prefixes`` is ordered by preference: [(source_name, [colH, colD, colA]), ...].
    """
    n = len(df)
    probs = pd.DataFrame(np.nan, index=df.index, columns=["h", "d", "a"])
    src = pd.Series([None] * n, index=df.index, dtype=object)
    for name, (ch, cd, ca) in prefixes:
        p = odds.implied_probs(_num(df, ch), _num(df, cd), _num(df, ca))
        fill = probs["h"].isna() & p["h"].notna()
        probs.loc[fill] = p.loc[fill]
        src[fill] = name
    return probs, src


def clean_season(
    raw: pd.DataFrame, start_year: int, div: str, source: str = "football-data.co.uk"
) -> pd.DataFrame:
    """Convert one raw season frame to the canonical match schema."""
    df = raw.rename(columns={k: v for k, v in ALIASES.items() if k in raw.columns}).copy()
    for needed in ("Date", "HomeTeam", "AwayTeam"):
        if needed not in df.columns:
            raise ValueError(f"missing column {needed}")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    date = parse_dates(df["Date"])
    check_season_window(date, start_year)

    season = config.season_label(start_year)
    home = df["HomeTeam"].map(canonical)
    away = df["AwayTeam"].map(canonical)
    hg, ag = _num(df, "FTHG"), _num(df, "FTAG")
    played = hg.notna() & ag.notna()

    out = pd.DataFrame(index=df.index)
    out["div"] = div
    out["season"] = season
    out["date"] = date
    out["kickoff"] = df["Time"] if "Time" in df.columns else None  # nullable pre-2019-20
    out["home_team"], out["away_team"] = home, away
    out["home_goals"], out["away_goals"] = hg, ag
    res = pd.Series(np.where(hg > ag, "H", np.where(hg < ag, "A", "D")), index=df.index)
    out["result"] = res.where(played)
    if "FTR" in df.columns:  # gate: the file's own result column must agree with its goals
        stated = df["FTR"].astype(str).str.strip().str.upper()
        bad = played & stated.isin(["H", "D", "A"]) & (stated != res)
        if bad.any():
            rows = df.loc[bad, ["Date", "HomeTeam", "AwayTeam"]].head(3).to_dict("records")
            raise ValueError(f"FTR contradicts the score in {int(bad.sum())} rows, e.g. {rows}")
    for src_col, dst in _STAT_MAP.items():
        out[dst] = _num(df, src_col)

    mkt, mkt_src = _market(
        df,
        [
            ("pinnacle", ("PSH", "PSD", "PSA")),
            ("average", ("AvgH", "AvgD", "AvgA")),
            ("average", ("BbAvH", "BbAvD", "BbAvA")),
        ],
        "mkt",
    )
    out[["p_mkt_h", "p_mkt_d", "p_mkt_a"]] = mkt.to_numpy()
    out["mkt_src"] = mkt_src
    cl, cl_src = _market(
        df,
        [
            ("pinnacle", ("PSCH", "PSCD", "PSCA")),
            ("average", ("AvgCH", "AvgCD", "AvgCA")),
        ],
        "close",
    )
    out[["p_close_h", "p_close_d", "p_close_a"]] = cl.to_numpy()
    out["close_src"] = cl_src

    over = _first(df, ["P>2.5", "Avg>2.5", "BbAv>2.5"])
    under = _first(df, ["P<2.5", "Avg<2.5", "BbAv<2.5"])
    out["p_over25"] = odds.two_way_prob(over, under)
    out["ah_line"] = _first(df, ["AHh", "BbAHh"])
    out["ah_close_line"] = _num(df, "AHCh")

    out["status"] = np.where(played, "played", "scheduled")
    out["source"] = source
    out["ingested_at"] = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    out["match_id"] = (
        season + "_" + out["date"].dt.strftime("%Y-%m-%d") + "_" + home + "_" + away
    )
    return out[RESULT_COLS]
