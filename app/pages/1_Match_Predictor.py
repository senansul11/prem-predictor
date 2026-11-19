"""Match Predictor: pick any two teams and see the predicted score and chances."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import OUTCOME_TEXT, footer, setup  # noqa: E402
from plpredictor.data.team_names import short_name  # noqa: E402
from plpredictor.models.dixon_coles import (  # noqa: E402
    matrix_summary,
    predicted_pick,
    score_matrices,
    top_scorelines,
)
from ui import ACCENT, AWAY, DRAW, versus_header  # noqa: E402

data = setup("Match Predictor", "🎯")
st.title("🎯 Match Predictor")
st.write("Pick a home team and an away team to see the predicted score and chances.")

teams = sorted(data.table.index)
rt = data.ratings.set_index("team")
c1, c2 = st.columns(2)
home = c1.selectbox("Home team", teams, index=teams.index("Arsenal"), format_func=short_name)
away_options = [t for t in teams if t != home]
default_away = "Chelsea" if "Chelsea" in away_options else away_options[0]
away = c2.selectbox(
    "Away team", away_options, index=away_options.index(default_away), format_func=short_name
)

p = data.params
lam = np.exp(p["mu0"] + p["home_adv"] + rt.loc[home, "attack"] - rt.loc[away, "defence"])
mu = np.exp(p["mu0"] + rt.loc[away, "attack"] - rt.loc[home, "defence"])
mat = score_matrices(np.array([lam]), np.array([mu]), p["rho"], int(p["max_goals"]))[0]
summ = matrix_summary(mat[None]).iloc[0]
pick = predicted_pick(mat)

# was / will this exact fixture be played this season?
played = data.results[(data.results["home_team"] == home) & (data.results["away_team"] == away)]
upcoming = data.pred[(data.pred["home_team"] == home) & (data.pred["away_team"] == away)]
if len(played):
    r = played.iloc[0]
    st.info(
        f"This fixture was already played on {r['date']:%d %b %Y}: "
        f"**{short_name(home)} {int(r['home_goals'])}–{int(r['away_goals'])} {short_name(away)}**. "
        "The prediction below is for a hypothetical replay."
    )
elif len(upcoming):
    d = upcoming.iloc[0]["date"]
    st.success(
        "This fixture is still to be played"
        + (f" on **{d:%a %d %b %Y, %H:%M}**." if str(d) != "NaT" else " (date to be confirmed).")
    )

st.markdown(versus_header(home, away), unsafe_allow_html=True)
winner = {"H": short_name(home) + " win", "A": short_name(away) + " win", "D": "Draw"}[pick["outcome"]]
m1, m2, m3 = st.columns(3)
m1.metric("Predicted score", f"{pick['home_goals']} – {pick['away_goals']}")
m2.metric("Predicted result", winner)
m3.metric("Confidence", f"{pick['confidence']} ({pick['outcome_prob']:.0%})")

# --- win / draw / loss bar
fig = go.Figure()
for _key, label, val, colour in (
    ("H", f"{short_name(home)} win", summ["p_h"], ACCENT),
    ("D", "Draw", summ["p_d"], DRAW),
    ("A", f"{short_name(away)} win", summ["p_a"], AWAY),
):
    fig.add_trace(
        go.Bar(
            y=[""], x=[val * 100], orientation="h", name=label, marker_color=colour,
            text=f"{label}<br>{val:.0%}", textposition="inside", insidetextanchor="middle",
            hovertemplate=f"{label}: {val:.1%}<extra></extra>",
        )
    )
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    barmode="stack", height=130, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(range=[0, 100], visible=False), yaxis=dict(visible=False),
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

left, right = st.columns(2)
with left:
    st.subheader("Most likely scores")
    rows = [
        {"Score": f"{h}–{a}", "Chance": f"{q:.1%}"} for h, a, q in top_scorelines(mat, 6)
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown(
        f"**Expected goals:** {short_name(home)} {summ['xg_h']:.2f} · {short_name(away)} {summ['xg_a']:.2f}  \n"
        f"**3+ goals in the match:** {summ['p_over25']:.0%} · **Both teams score:** {summ['p_btts']:.0%}"
    )
with right:
    st.subheader("Scoreline heatmap")
    g = 6
    sub = mat[:g, :g] * 100
    heat = px.imshow(
        sub, x=[str(i) for i in range(g)], y=[str(i) for i in range(g)],
        labels=dict(x=f"{short_name(away)} goals", y=f"{short_name(home)} goals", color="%"),
        text_auto=".1f", color_continuous_scale=[[0, "#10173A"], [0.5, "#6D28D9"], [1, "#00E5A8"]], aspect="equal",
    )
    heat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(heat, width="stretch", config={"displayModeBar": False})
    st.caption("Each cell is the chance (%) of that exact score. Rows: home goals, columns: away goals.")

with st.expander("How is this calculated?"):
    st.markdown(
        "Every team has an **attack** strength (how well it scores) and a **defence** strength (how well "
        "it stops goals), worked out from past results, with recent matches counting for more. For this "
        "match we combine the two teams' strengths, add a small boost for playing at home, and work out "
        "how likely each possible score is. Adding up all the scores where the home side wins, where it's "
        "level, and where the away side wins gives the three percentages."
    )
footer(data)
_ = OUTCOME_TEXT
