"""Reliability chart of the final model on the test seasons (replaces the older chart).

Run: python -m plpredictor.evaluation.calibration_plot
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plpredictor import config  # noqa: E402
from plpredictor.data import build_dataset  # noqa: E402
from plpredictor.evaluation import backtest, baselines, metrics  # noqa: E402
from plpredictor.evaluation.dc_eval import make_dc_fp  # noqa: E402
from plpredictor.models.dixon_coles import DCConfig  # noqa: E402


def main() -> None:
    m = build_dataset.load_matches()
    cfg = DCConfig(**json.loads((config.OUTPUTS_DIR / "dc_config.json").read_text()))
    p = backtest.walk_forward(make_dc_fp(cfg), m, backtest.TEST_SEASONS, refit_days=7)
    close = baselines.market_preds(m, "close").loc[p.index]
    ok = close.notna().all(axis=1)
    y = metrics.outcome_index(m.loc[p.index[ok], "result"])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor("#090D1F")
    ax.set_facecolor("#090D1F")
    ax.plot([0, 1], [0, 1], ":", color="#9AA7D1", lw=1)
    for label, probs, colour in (
        ("Our predictor", p[ok].to_numpy(), "#00E5A8"),
        ("Bookmakers", close[ok].to_numpy(), "#FF7A59"),
    ):
        cb = metrics.calibration_bins(probs, y, outcome=0, bins=8)
        ax.plot(cb["mean_pred"], cb["observed"], "o-", color=colour, label=label, lw=2)
    ax.set_xlabel("What we said: chance of a home win", color="#F1F5FF")
    ax.set_ylabel("How often it really happened", color="#F1F5FF")
    ax.tick_params(colors="#8B98A5")
    for spine in ax.spines.values():
        spine.set_color("#2A3668")
    ax.legend(facecolor="#131A3A", edgecolor="#2A3668", labelcolor="#F1F5FF")
    fig.tight_layout()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.REPORTS_DIR / "calibration.png", dpi=130, facecolor=fig.get_facecolor())
    print("saved", config.REPORTS_DIR / "calibration.png", "n =", int(ok.sum()))


if __name__ == "__main__":
    main()
