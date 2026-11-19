"""CLI: ``python -m plpredictor.pipeline build [--download]``."""

from __future__ import annotations

import argparse

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.data.sources import football_data_uk as fduk


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="plpredictor.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="clean cached CSVs into matches.parquet")
    b.add_argument("--download", action="store_true", help="fetch missing season CSVs first")
    sub.add_parser("fixtures", help="download the remaining schedule (needs API key in .env)")
    rf = sub.add_parser("refresh", help="daily update: download, validate, re-predict")
    rf.add_argument("--force", action="store_true", help="re-run even if nothing changed")
    sim = sub.add_parser("simulate", help="fit model, simulate the season, write outputs")
    sim.add_argument("--sims", type=int, default=config.N_SIMS)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        if args.download:
            for div in (config.LEAGUE, config.SECOND_TIER):
                for start in config.season_starts():
                    fduk.download_season(start, div)
        df = build_dataset.build()
        played = df[(df["div"] == config.LEAGUE) & (df["status"] == "played")]
        print(f"matches.parquet: {len(df)} rows; played E0 rows: {len(played)}")
        print(build_dataset.COVERAGE_PATH.read_text())

    elif args.cmd == "fixtures":
        from plpredictor.data.sources import fixtures_api

        n = len(fixtures_api.fetch_upcoming())
        print(f"cached {n} upcoming fixtures")
    elif args.cmd == "refresh":
        from plpredictor import refresh

        refresh.main(force=args.force)
    elif args.cmd == "simulate":
        from plpredictor.simulation import run

        out = run.run(args.sims)
        print(out["meta"])
        print(out["summary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
