"""How accurate is it? Plain-language track record and how the predictions are made."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import OUT, REPORTS, footer, setup  # noqa: E402

data = setup("How Accurate Is It?", "📈")
st.title("📈 How Accurate Is It?")

st.markdown(
    """
We tested the predictor on **2,000 real Premier League matches** (from 2021 to now). For every
match it only used information that was available **before kick-off**, exactly as it would have
on the day. Then we checked how often it picked the right result.
"""
)

comp = pd.read_csv(OUT / "improve_comparison.csv")
comp = comp[(comp["period"] == "test 2021-27") & comp["model"].str.startswith("FINAL")]
base = pd.read_csv(OUT / "baselines.csv")
base = base[base["period"] == "test 2021-27"].set_index("model")
ours = float(comp.iloc[0]["accuracy"])

rows = [
    ("Always pick the home team", float(base.loc["base rates", "accuracy"]), "#64748B"),
    ("Our predictor", ours, "linear-gradient(90deg,#8B5CF6,#00E5A8)"),
    ("Bookmakers' odds, days before", float(base.loc["market pre-closing", "accuracy"]), "#FFC857"),
    ("Bookmakers' odds, at kick-off", float(base.loc["market closing", "accuracy"]), "#FF7A59"),
]
lo = 0.30  # bars start at 30% so differences are visible
html = ""
for label, val, colour in rows:
    width = max((val - lo) / (0.60 - lo), 0.02) * 100
    html += (
        f'<div class="score-row"><div class="lab">{label}</div>'
        f'<div class="track"><div class="fill" style="width:{width:.0f}%;background:{colour}"></div></div>'
        f'<div class="val">{val * 100:.0f}%</div></div>'
    )
st.subheader("How often the top pick was right")
st.markdown(html, unsafe_allow_html=True)
st.caption("“Right” means we called the correct result: home win, draw or away win. Bars start at 30%.")

st.info(
    f"**About {round(ours * 100)} in 100 results called correctly.** That is far better than just backing "
    "the home side, and just a little behind the bookmakers, who also know about injuries, "
    "line-ups and team news that we can't see. Football is full of upsets, so no one gets close to 100%."
)

st.subheader("Do the percentages match reality?")
st.write(
    "When we say a team has a 60% chance, it should win about 6 times out of 10. "
    "The chart compares our percentages with what really happened, next to the bookmakers'. "
    "The closer a line is to the dotted diagonal, the more trustworthy the percentages."
)
cal = REPORTS / "calibration.png"
if cal.exists():
    st.image(str(cal), width=440)

st.subheader("What about the final-table predictions?")
st.markdown(
    "We also replayed the last five seasons from 10 and 20 games in, and compared our predicted "
    "title, top-4 and relegation chances with what happened. They were sensible but not sharp: "
    "a team given 70% to win the league still misses out about 3 times in 10."
)

st.subheader("How the predictions are made")
st.markdown(
    """
1. **Past results.** We use every Premier League and Championship match since 2015, including the
   scores and shots on target.
2. **Team strengths.** From those results we work out how good each team is at scoring and at
   stopping goals. Recent matches count for more, and home advantage is included.
3. **Match chances.** For each upcoming game we work out the chance of every possible scoreline.
   Adding them up gives the chances of a home win, draw or away win.
4. **The rest of the season.** We play out all remaining matches **10,000 times** and see where
   every team finishes, which gives the predicted table.
5. **What it can't know.** Injuries, suspensions, line-ups, transfers, managers and motivation.
   Treat it as an informed guide, not a crystal ball.
"""
)
footer(data)
