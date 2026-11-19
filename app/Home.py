"""Prem Predictor home page: predicted results and predicted final table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import call_text, footer, headline_accuracy, setup  # noqa: E402
from plpredictor.data.team_names import short_name  # noqa: E402
from ui import cards, hero, predicted_table  # noqa: E402

data = setup("Predicted Results", "⚽")
pred, meta = data.pred, data.meta

acc = headline_accuracy()
stats = [
    ("Played", str(meta["matches_played"])),
    ("To go", str(meta["fixtures_remaining"])),
    ("Right result", f"{acc * 100:.0f}%" if acc else "–"),
    ("Data as of", pd.Timestamp(meta["data_as_of"]).strftime("%d %b")),
]
st.markdown(
    hero(
        "Premier League <span>Predictor</span>",
        f"Season {meta['season']} · predicted scores for every remaining match",
        stats,
    ),
    unsafe_allow_html=True,
)

with st.expander("How to read the predictions", expanded=False):
    st.markdown(
        """
- **Predicted score** is the most likely scoreline *for the predicted result*. Even the best
  single score is only about 10–13% likely, so treat it as a best guess, not a promise.
- **The percentage** is our chance that the predicted result happens. The bar under each
  match splits the chances: teal = home win, grey = draw, amber = away win.
- **Strong pick** means 55% or more, **Lean** 45–55%, **Toss-up** below 45%.
- We rarely pick a draw as the single most likely result, even though about 1 game in 4
  ends level. That is normal: a draw is almost never the *most* likely of the three outcomes.
- In our tests on past seasons we called the right result in about **54 of every 100 matches**.
        """
    )

tab_fix, tab_table = st.tabs(["📅 Predicted results", "🏆 Predicted final table"])

# ---------------------------------------------------------------- predicted results
with tab_fix:
    dated_all = pred[pred["date"].notna()]
    nxt = dated_all["game_week"].min() if len(dated_all) else None
    if nxt is not None and not np.isnan(nxt):
        this_round = dated_all[dated_all["game_week"] == nxt]
        top = this_round.sort_values("pred_outcome_prob", ascending=False).head(4)
        st.subheader(f"⭐ Top picks · Gameweek {int(nxt)}")
        st.markdown(cards(top), unsafe_allow_html=True)
        st.write("")

    st.subheader("All upcoming matches")
    teams = sorted(data.table.index)
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    picked = c1.multiselect("Show only these teams", teams, format_func=short_name)
    conf = c2.multiselect(
        "Confidence",
        ["Strong pick", "Lean", "Toss-up"],
        default=["Strong pick", "Lean", "Toss-up"],
    )
    n_rows = c3.selectbox("Matches shown", [10, 20, 50, 280], index=1)
    view = c4.radio("View", ["Cards", "Table"], horizontal=True)

    df = pred[pred["confidence"].isin(conf)].copy()
    if picked:
        df = df[df["home_team"].isin(picked) | df["away_team"].isin(picked)]
    dated = df[df["date"].notna()].sort_values("date").head(n_rows)
    undated = df[df["date"].isna()].sort_values(["home_team", "away_team"])

    def as_table(frame: pd.DataFrame) -> pd.DataFrame:
        pct = lambda col: (frame[col] * 100).round(0).astype(int).astype(str)  # noqa: E731
        return pd.DataFrame(
            {
                "Date": frame["date"].dt.strftime("%a %d %b, %H:%M").fillna("TBC"),
                "Match": frame["home_team"].map(short_name) + " v " + frame["away_team"].map(short_name),
                "Predicted score": frame["pred_score"],
                "Result": frame.apply(call_text, axis=1),
                "Chance": pct("pred_outcome_prob") + "%",
                "Confidence": frame["confidence"],
                "H / D / A %": pct("p_h") + " / " + pct("p_d") + " / " + pct("p_a"),
            }
        ).reset_index(drop=True)

    def show(frame: pd.DataFrame) -> None:
        if view == "Cards":
            st.markdown(cards(frame), unsafe_allow_html=True)
        else:
            st.dataframe(as_table(frame), hide_index=True, width="stretch")

    if dated.empty:
        st.info("No matches match these filters.")
    else:
        show(dated)
    if len(undated):
        with st.expander(f"Date to be confirmed ({len(undated)} matches, schedule not yet published)"):
            show(undated.head(n_rows))
    st.caption("Tip: open **Match Predictor** in the sidebar for full scoreline odds on any pairing.")

# ---------------------------------------------------------------- predicted table
with tab_table:
    st.subheader("Where we expect each team to finish")
    s = data.summary.merge(data.table[["played"]], left_on="team", right_index=True, suffixes=("", "_t"))
    pos = data.positions
    cum = pos.cumsum(axis=1).to_numpy()
    p10 = (cum >= 0.10).argmax(axis=1) + 1
    p90 = (cum >= 0.90).argmax(axis=1) + 1
    rng = pd.Series(
        [f"{a}–{b}" if a != b else f"{a}" for a, b in zip(p10, p90, strict=True)], index=pos.index
    )
    now_rank = {
        t: i + 1
        for i, t in enumerate(data.table.sort_values(["pts", "gd", "gf"], ascending=False).index)
    }
    s = s.sort_values(["exp_points", "exp_gd"], ascending=False).reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "pos": np.arange(1, len(s) + 1),
            "team": s["team"],
            "played": s["played"].astype(int),
            "pts_now": s["pts_now"].astype(int),
            "pred_pts": s["exp_points"].round(0).astype(int),
            "pred_gd": s["exp_gd"].round(0).astype(int),
            "move": [now_rank[t] - (i + 1) for i, t in enumerate(s["team"])],
            "rng": s["team"].map(rng),
        }
    )
    st.markdown(predicted_table(rows), unsafe_allow_html=True)
    st.caption(
        "Predicted points are averages over 10,000 plays of the rest of the season, so they are rounded and ties are "
        "broken by predicted goal difference. “Likely finish” is where the team ends up in the middle 80% "
        "of those plays. “vs now” compares the predicted position with today's table."
    )
    with st.expander("Title, top-4 and relegation chances"):
        chances = pd.DataFrame(
            {
                "Team": s["team"].map(short_name),
                "Win title": (s["p_title"] * 100).round(1),
                "Top 4": (s["p_top4"] * 100).round(1),
                "Relegated": (s["p_relegation"] * 100).round(1),
            }
        )
        st.dataframe(
            chances,
            hide_index=True,
            width="stretch",
            column_config={
                "Win title": st.column_config.NumberColumn(format="%.1f%%"),
                "Top 4": st.column_config.NumberColumn(format="%.1f%%"),
                "Relegated": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

footer(data)
