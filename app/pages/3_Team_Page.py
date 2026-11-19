"""Team Page: form, strength, remaining fixtures and finishing-position chances."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import footer, setup  # noqa: E402
from plpredictor.data.team_names import short_name  # noqa: E402
from ui import team_header  # noqa: E402

data = setup("Team Page", "🛡️")
st.title("🛡️ Team Page")

teams = sorted(data.table.index)
team = st.selectbox("Team", teams, index=teams.index("Arsenal"), format_func=short_name)

# ---- current standing
ranked = data.table.sort_values(["pts", "gd", "gf"], ascending=False)
pos_now = list(ranked.index).index(team) + 1
row = data.table.loc[team]
s = data.summary.set_index("team").loc[team]
st.markdown(
    team_header(team, f"Position {pos_now} · {int(row['pts'])} points from {int(row['played'])} games"),
    unsafe_allow_html=True,
)
m = st.columns(4)
m[0].metric("League position now", f"{pos_now}")
m[1].metric("Points (played)", f"{int(row['pts'])} ({int(row['played'])})")
m[2].metric("Predicted final points", f"{s['exp_points']:.0f}")
m[3].metric("Predicted finish", f"{s['mean_position']:.1f}")

# ---- recent form
res = data.results
mine = res[(res["home_team"] == team) | (res["away_team"] == team)].sort_values("date")
form_rows = []
for r in mine.itertuples():
    home = r.home_team == team
    gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
    outcome = "W" if gf > ga else "D" if gf == ga else "L"
    form_rows.append(
        {
            "Date": r.date.strftime("%d %b"),
            "Venue": "H" if home else "A",
            "Opponent": short_name(r.away_team if home else r.home_team),
            "Score": f"{int(gf)}–{int(ga)}",
            "Result": outcome,
        }
    )
form = pd.DataFrame(form_rows)
st.subheader("Recent form")
if form.empty:
    st.write("No matches played yet.")
else:
    chips = {"W": "🟩", "D": "⬜", "L": "🟥"}
    st.markdown("Last 5 (oldest → newest): " + " ".join(chips[x] for x in form["Result"].tail(5)))
    st.dataframe(form.tail(10).iloc[::-1], hide_index=True, width="stretch")

# ---- strength
rt = data.ratings.set_index("team")
overall = (rt["attack"] + rt["defence"]).rank(ascending=False).astype(int)
att = rt["attack"].rank(ascending=False).astype(int)
dfn = rt["defence"].rank(ascending=False).astype(int)
st.subheader("Team strength")
c = st.columns(3)
c[0].metric("Overall rank (of 20)", int(overall[team]))
c[1].metric("Attack rank", int(att[team]))
c[2].metric("Defence rank", int(dfn[team]))
st.caption("Out of 20 teams, based on goals scored and conceded in recent matches (1 = strongest).")

# ---- remaining fixtures
st.subheader("Remaining fixtures")
rem = data.pred[(data.pred["home_team"] == team) | (data.pred["away_team"] == team)].copy()
opp_strength = []
rows = []
for r in rem.itertuples():
    home = r.home_team == team
    opp = r.away_team if home else r.home_team
    win_p = r.p_h if home else r.p_a
    score = r.pred_score if home else "-".join(reversed(r.pred_score.split("-")))
    exp_pts = 3 * win_p + r.p_d
    opp_strength.append(overall[opp])
    rows.append(
        {
            "_d": r.date if pd.notna(r.date) else pd.Timestamp.max,
            "Date": r.date.strftime("%a %d %b") if pd.notna(r.date) else "TBC",
            "Venue": "Home" if home else "Away",
            "Opponent": short_name(opp),
            "Predicted score": score,
            "Win %": round(win_p * 100),
            "Draw %": round(r.p_d * 100),
            "Loss %": round((r.p_a if home else r.p_h) * 100),
            "Expected pts": round(exp_pts, 2),
            "Opponent rank": int(overall[opp]),
        }
    )
fx = pd.DataFrame(rows).sort_values("_d").drop(columns="_d")
st.dataframe(fx, hide_index=True, width="stretch", height=420)
if len(fx):
    avg = pd.Series(opp_strength).mean()
    st.caption(
        f"Average opponent rank: **{avg:.1f}** (10.5 is average; lower means a harder run-in). "
        f"Expected points from the remaining matches: **{fx['Expected pts'].sum():.1f}**."
    )

# ---- finishing positions
st.subheader("Finishing position chances")
dist = data.positions.loc[team] * 100
fig = go.Figure(
    go.Bar(
        x=list(range(1, 21)), y=dist.values,
        marker_color=["#00E5A8"] * 4 + ["#00C2FF"] + ["#3B4A8C"] * 12 + ["#FF4D8D"] * 3,
        hovertemplate="Position %{x}: %{y:.1f}%<extra></extra>",
    )
)
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Final league position",
    yaxis_title="Chance (%)", xaxis=dict(dtick=1),
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Teal: top 4 · Blue: 5th · Pink: relegation. Based on playing out the rest of the season 10,000 times.")
footer(data)
