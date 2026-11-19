"""Visual building blocks: CSS theme, team badges, match cards, hero, tables (HTML)."""

from __future__ import annotations

import html

import pandas as pd

from plpredictor.data.team_names import short_name

# --- palette (vivid dark: navy base, violet / electric-teal / coral accents)
BG = "#090D1F"
CARD = "#131A3A"
BORDER = "#2A3668"
TEXT = "#F1F5FF"
MUTED = "#9AA7D1"
ACCENT = "#00E5A8"  # mint-teal: home / positive
DRAW = "#A78BFA"  # violet
AWAY = "#FF7A59"  # coral
GOOD, WARN, BAD = "#00E5A8", "#FFC857", "#FF4D8D"

CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}}
[data-testid="stMetricValue"] > div {{white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.2;}}
[data-testid="stMetricValue"] {{font-size: 1.55rem;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.hero {{background: linear-gradient(125deg, #6D28D9 0%, #2563EB 50%, #06B6D4 100%);
  border-radius: 20px; padding: 24px 28px; margin-bottom: 20px;
  box-shadow: 0 12px 40px rgba(99,102,241,.38);}}
.hero-title {{font-size: 2rem; font-weight: 850; line-height: 1.15; color: #fff;}}
.hero-title span {{color: #FDE047; text-shadow: 0 0 18px rgba(253,224,71,.45);}}
.hero-sub {{color: rgba(255,255,255,.82); margin-top: 4px; font-size: .96rem;}}
.stats {{display: flex; flex-wrap: wrap; gap: 14px 36px; margin-top: 18px;}}
.stats div {{display: flex; flex-direction: column;}}
.stats b {{font-size: 1.7rem; font-weight: 850; color: #fff;}}
.stats span {{color: rgba(255,255,255,.75); font-size: .74rem; text-transform: uppercase; letter-spacing: .08em;}}
.cards {{display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px;}}
.mc {{position: relative; overflow: hidden; background: linear-gradient(160deg, #18214F 0%, #10173A 100%);
  border: 1px solid {BORDER}; border-radius: 18px; padding: 16px 18px 14px;
  box-shadow: 0 6px 22px rgba(0,0,0,.35);}}
.mc::before {{content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px;
  background: linear-gradient(90deg, #00E5A8, #8B5CF6, #FF7A59);}}
.mc-top {{display: flex; justify-content: space-between; align-items: center; color: {MUTED};
  font-size: .78rem; margin: 4px 0 12px;}}
.mc-main {{display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px;}}
.mc-team {{display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: .96rem; color: #fff;}}
.mc-team.right {{justify-content: flex-end; text-align: right;}}
.mc-score {{font-size: 2rem; font-weight: 900; letter-spacing: .02em; padding: 0 6px; color: #fff;
  text-shadow: 0 0 22px rgba(139,92,246,.75);}}
.bar {{display: flex; height: 10px; border-radius: 8px; overflow: hidden; margin: 14px 0 9px; background: #0B1030;}}
.bar i {{display: block; height: 100%;}}
.bar i.h {{background: linear-gradient(90deg, #00E5A8, #00C2FF);}}
.bar i.d {{background: #A78BFA;}}
.bar i.a {{background: linear-gradient(90deg, #FF7A59, #FF4D8D);}}
.mc-foot {{display: flex; justify-content: space-between; font-size: .9rem; font-weight: 700; color: #fff;}}
.mc-sub {{color: {MUTED}; font-size: .74rem; margin-top: 3px;}}
.pill {{padding: 3px 11px; border-radius: 999px; font-size: .7rem; font-weight: 800; letter-spacing: .03em;}}
.pill.strong {{background: #00E5A8; color: #03261D; box-shadow: 0 0 14px rgba(0,229,168,.45);}}
.pill.lean {{background: #FFC857; color: #3A2700; box-shadow: 0 0 14px rgba(255,200,87,.35);}}
.pill.toss {{background: rgba(167,139,250,.28); color: #E2D9FF; border: 1px solid rgba(167,139,250,.55);}}
.badge {{display: inline-flex; align-items: center; justify-content: center; border-radius: 50%;
  font-weight: 800; letter-spacing: .02em; flex: none; box-shadow: 0 0 0 2px rgba(255,255,255,.22);}}
.tt {{width: 100%; border-collapse: collapse; font-size: .92rem;}}
.tt th {{text-align: left; color: {MUTED}; font-weight: 700; font-size: .72rem; text-transform: uppercase;
  letter-spacing: .06em; padding: 9px 10px; border-bottom: 2px solid {BORDER};}}
.tt td {{padding: 10px 10px; border-bottom: 1px solid #1E2750; vertical-align: middle; color: #fff;}}
.tt tr:hover td {{background: rgba(139,92,246,.10);}}
.tt tr.z-cl td:first-child {{box-shadow: inset 4px 0 0 #00E5A8;}}
.tt tr.z-eu td:first-child {{box-shadow: inset 4px 0 0 #00C2FF;}}
.tt tr.z-rel td:first-child {{box-shadow: inset 4px 0 0 #FF4D8D;}}
.tt .num {{text-align: right; font-variant-numeric: tabular-nums;}}
.ptsbar {{display: flex; align-items: center; gap: 8px; justify-content: flex-end;}}
.ptsbar div {{height: 7px; border-radius: 5px; background: linear-gradient(90deg, #8B5CF6, #00E5A8);}}
.up {{color: #00E5A8; font-weight: 700;}} .down {{color: #FF4D8D; font-weight: 700;}} .same {{color: {MUTED};}}
.wrap {{overflow-x: auto;}}
.legend {{color: {MUTED}; font-size: .82rem; margin-top: 10px;}}
.dot {{display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px;}}
.score-row {{display: grid; grid-template-columns: minmax(150px, 1.3fr) 3fr auto; gap: 12px; align-items: center;
  margin: 12px 0;}}
.score-row .lab {{font-weight: 700; color: #fff;}}
.score-row .track {{height: 16px; border-radius: 10px; background: #0B1030; overflow: hidden;}}
.score-row .fill {{height: 100%; border-radius: 10px;}}
.score-row .val {{font-weight: 850; font-size: 1.1rem; min-width: 56px; text-align: right; color: #fff;}}
@media (max-width: 640px) {{
  .block-container {{padding-left: 1rem; padding-right: 1rem;}}
  .hero {{padding: 18px;}} .hero-title {{font-size: 1.5rem;}} .stats b {{font-size: 1.3rem;}}
  .cards {{grid-template-columns: 1fr;}}
  .mc-score {{font-size: 1.6rem;}}
  .tt {{font-size: .82rem;}} .tt td, .tt th {{padding: 8px 6px;}}
  .score-row {{grid-template-columns: 1fr auto; }} .score-row .track {{grid-column: 1 / -1; order: 3;}}
}}
</style>
"""

# --- club colours (kit-inspired; used for small initials badges, not crests)
CLUBS: dict[str, tuple[str, str, str]] = {  # canonical -> (abbr, bg, fg)
    "Arsenal": ("ARS", "#EF0107", "#FFFFFF"),
    "Aston Villa": ("AVL", "#7B1E48", "#95BFE5"),
    "AFC Bournemouth": ("BOU", "#DA291C", "#000000"),
    "Brentford": ("BRE", "#E30613", "#FFFFFF"),
    "Brighton & Hove Albion": ("BHA", "#0057B8", "#FFFFFF"),
    "Chelsea": ("CHE", "#034694", "#FFFFFF"),
    "Coventry City": ("COV", "#59CBE8", "#0B2239"),
    "Crystal Palace": ("CRY", "#1B458F", "#E4002B"),
    "Everton": ("EVE", "#003399", "#FFFFFF"),
    "Fulham": ("FUL", "#FFFFFF", "#000000"),
    "Hull City": ("HUL", "#F5A12D", "#000000"),
    "Ipswich Town": ("IPS", "#3A64A3", "#FFFFFF"),
    "Leeds United": ("LEE", "#FFCD00", "#1D428A"),
    "Liverpool": ("LIV", "#C8102E", "#FFFFFF"),
    "Manchester City": ("MCI", "#6CABDD", "#0B1F3A"),
    "Manchester United": ("MUN", "#DA291C", "#FBE122"),
    "Newcastle United": ("NEW", "#E5E7EB", "#111111"),
    "Nottingham Forest": ("NFO", "#DD0000", "#FFFFFF"),
    "Sunderland": ("SUN", "#EB172B", "#FFFFFF"),
    "Tottenham Hotspur": ("TOT", "#132257", "#FFFFFF"),
}


def badge(team: str, size: int = 30) -> str:
    """Round club-coloured badge with the 3-letter abbreviation."""
    abbr, bg, fg = CLUBS.get(team, (team[:3].upper(), "#475569", "#FFFFFF"))
    font = max(int(size * 0.34), 8)
    return (
        f'<span class="badge" style="width:{size}px;height:{size}px;background:{bg};color:{fg};'
        f'font-size:{font}px;">{abbr}</span>'
    )


def name(team: str) -> str:
    return html.escape(short_name(team))


def _pct(x: float) -> int:
    return int(round(x * 100))


def hero(title_html: str, subtitle: str, stats: list[tuple[str, str]]) -> str:
    items = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in stats)
    return (
        f'<div class="hero"><div class="hero-title">{title_html}</div>'
        f'<div class="hero-sub">{subtitle}</div><div class="stats">{items}</div></div>'
    )


def match_card(row: pd.Series) -> str:
    """One fixture as a card: teams, predicted score, win/draw/loss bar, confidence."""
    when = row["date"].strftime("%a %d %b · %H:%M") if pd.notna(row["date"]) else "Date TBC"
    cls = {"Strong pick": "strong", "Lean": "lean", "Toss-up": "toss"}[row["confidence"]]
    h, d, a = _pct(row["p_h"]), _pct(row["p_d"]), _pct(row["p_a"])
    hg, ag = row["pred_score"].split("-")
    call = {
        "H": f"{short_name(row['home_team'])} win",
        "A": f"{short_name(row['away_team'])} win",
        "D": "Draw",
    }[row["pred_outcome"]]
    return (
        '<div class="mc">'
        f'<div class="mc-top"><span>{when}</span><span class="pill {cls}">{row["confidence"]}</span></div>'
        '<div class="mc-main">'
        f'<div class="mc-team">{badge(row["home_team"])}<span>{name(row["home_team"])}</span></div>'
        f'<div class="mc-score">{hg}&nbsp;–&nbsp;{ag}</div>'
        f'<div class="mc-team right"><span>{name(row["away_team"])}</span>{badge(row["away_team"])}</div>'
        "</div>"
        f'<div class="bar"><i class="h" style="width:{h}%"></i>'
        f'<i class="d" style="width:{d}%"></i><i class="a" style="width:{a}%"></i></div>'
        f'<div class="mc-foot"><span>{html.escape(call)}</span><span>{_pct(row["pred_outcome_prob"])}%</span></div>'
        f'<div class="mc-sub">Home {h}% · Draw {d}% · Away {a}%</div>'
        "</div>"
    )


def cards(frame: pd.DataFrame) -> str:
    return '<div class="cards">' + "".join(match_card(r) for _, r in frame.iterrows()) + "</div>"


def predicted_table(rows: pd.DataFrame) -> str:
    """HTML league table. Expects columns: pos, team, played, pts_now, pred_pts, pred_gd, move, rng."""
    top = max(float(rows["pred_pts"].max()), 1.0)
    out = [
        '<div class="wrap"><table class="tt"><thead><tr><th>#</th><th>Team</th><th class="num">Pld</th>'
        '<th class="num">Pts now</th><th class="num">Predicted pts</th><th class="num">GD</th>'
        '<th class="num">vs now</th><th class="num">Likely finish</th></tr></thead><tbody>'
    ]
    for r in rows.itertuples():
        zone = "z-cl" if r.pos <= 4 else "z-eu" if r.pos == 5 else "z-rel" if r.pos > 17 else ""
        if r.move > 0:
            mv = f'<span class="up">▲ {r.move}</span>'
        elif r.move < 0:
            mv = f'<span class="down">▼ {-r.move}</span>'
        else:
            mv = '<span class="same">–</span>'
        width = int(60 * r.pred_pts / top)
        out.append(
            f'<tr class="{zone}"><td>{r.pos}</td>'
            f'<td><span style="display:inline-flex;align-items:center;gap:9px">{badge(r.team, 24)}'
            f"<b>{name(r.team)}</b></span></td>"
            f'<td class="num">{r.played}</td><td class="num">{r.pts_now}</td>'
            f'<td class="num"><span class="ptsbar"><div style="width:{width}px"></div><b>{r.pred_pts}</b></span></td>'
            f'<td class="num">{r.pred_gd:+d}</td><td class="num">{mv}</td><td class="num">{r.rng}</td></tr>'
        )
    out.append("</tbody></table></div>")
    out.append(
        '<div class="legend"><span class="dot" style="background:#00E5A8"></span>Champions League (top 4) &nbsp; '
        '<span class="dot" style="background:#00C2FF"></span>Europe (5th) &nbsp; '
        '<span class="dot" style="background:#FF4D8D"></span>Relegation (bottom 3)</div>'
    )
    return "".join(out)


def versus_header(home: str, away: str) -> str:
    return (
        '<div class="hero" style="padding:18px 22px"><div class="mc-main" style="grid-template-columns:1fr auto 1fr">'
        f'<div class="mc-team" style="font-size:1.1rem">{badge(home, 44)}<span>{name(home)}</span></div>'
        '<div class="hero-sub" style="font-weight:700">VS</div>'
        f'<div class="mc-team right" style="font-size:1.1rem"><span>{name(away)}</span>{badge(away, 44)}</div>'
        "</div></div>"
    )


def team_header(team: str, line: str) -> str:
    return (
        '<div class="hero" style="padding:18px 22px"><div class="mc-team" style="font-size:1.5rem;gap:14px">'
        f'{badge(team, 52)}<div><div class="hero-title" style="font-size:1.6rem">{name(team)}</div>'
        f'<div class="hero-sub">{line}</div></div></div></div>'
    )
