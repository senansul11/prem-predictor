"""What-If Simulator: lock in results for upcoming matches and see how the table changes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import fixture_label, footer, setup  # noqa: E402
from plpredictor import config  # noqa: E402
from plpredictor.data.team_names import short_name  # noqa: E402
from plpredictor.simulation import season_sim  # noqa: E402

data = setup("What-If Simulator", "🔮")
st.title("🔮 What-If Simulator")
st.write(
    "Choose upcoming matches, decide how they finish, and re-run the rest of the season "
    "to see how the final table changes."
)

pred = data.pred.reset_index(drop=True)
labels = {i: fixture_label(r) for i, r in pred.iterrows()}
order = pred.sort_values("date", na_position="last").index.tolist()

picked = st.multiselect(
    "Matches to lock in (up to 12)", order, format_func=lambda i: labels[i], max_selections=12
)
locked: dict[int, tuple[int, int]] = {}
for i in picked:
    r = pred.loc[i]
    c0, c1, c2 = st.columns([3, 1, 1])
    c0.markdown(f"**{labels[i]}**  \nOur pick: {r['pred_score']}")
    h, a = r["pred_score"].split("-")
    gh = c1.number_input(f"{short_name(r['home_team'])} goals", 0, 10, int(h), key=f"h{i}")
    ga = c2.number_input(f"{short_name(r['away_team'])} goals", 0, 10, int(a), key=f"a{i}")
    locked[i] = (int(gh), int(ga))

n_sims = st.radio("How many times to play out the season", [1000, 10000], index=1, horizontal=True)


@st.cache_data(show_spinner=False)
def run_sim(version: str, locks: tuple, n: int) -> pd.DataFrame:
    res = season_sim.simulate(
        data.table, pred, data.matrices, n_sims=n, seed=config.SEED, locked=dict(locks) or None
    )
    return res.summary().set_index("team")


if st.button("Run what-if", type="primary"):
    key = tuple(sorted(locked.items()))
    with st.spinner("Playing out the rest of the season…"):
        base = run_sim(data.meta["generated_at"], (), n_sims)
        after = run_sim(data.meta["generated_at"], key, n_sims)
    st.session_state["whatif"] = (base, after, key)

def change_colour(value: float) -> str:
    """Green for teams that gain points, pink for teams that lose them (no extra libraries)."""
    strength = min(abs(value) / 5, 1.0)
    if strength < 0.02:
        return ""
    rgb = "0,229,168" if value > 0 else "255,77,141"
    return f"background-color: rgba({rgb},{0.15 + 0.55 * strength:.2f}); color: white"


if "whatif" in st.session_state:
    base, after, key = st.session_state["whatif"]
    base = base.reindex(after.index)  # same team order in both tables
    if not key:
        st.info("No matches locked, so 'after' equals 'before'. Pick some matches above first.")
    st.subheader("Change in the predicted table")
    out = pd.DataFrame(
        {
            "Team": [short_name(t) for t in after.index],
            "Pts before": base["exp_points"].round(1).values,
            "Pts after": after["exp_points"].round(1).values,
            "Change (pts)": (after["exp_points"] - base["exp_points"]).round(1).values,
            "Title % before": (base["p_title"] * 100).round(1).values,
            "Title % after": (after["p_title"] * 100).round(1).values,
            "Top 4 % after": (after["p_top4"] * 100).round(1).values,
            "Relegation % before": (base["p_relegation"] * 100).round(1).values,
            "Relegation % after": (after["p_relegation"] * 100).round(1).values,
        }
    ).sort_values("Pts after", ascending=False)
    one_dp = {c: "{:.1f}" for c in out.columns if c != "Team"}
    st.dataframe(
        out.style.format(one_dp).map(change_colour, subset=["Change (pts)"]),
        hide_index=True,
        width="stretch",
        height=760,
    )
    st.caption(
        "“Before” and “after” use exactly the same method, so the only difference is the results you "
        "locked in. Numbers can differ slightly from the Home page, because this page uses a quicker version."
    )
footer(data)
_ = np
