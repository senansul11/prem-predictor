"""Daily refresh: download -> detect change -> validate -> rebuild predictions.

Steps (spec section 8.2):
1. Download the current-season CSVs (and, if an API key is set, the upcoming fixtures).
2. Hash them. If nothing changed since the last run, stop (no commit, no redeploy).
3. Save dated raw snapshots, rebuild the dataset through the validation gates.
4. Refit the model, re-run the season simulation, write ``data/outputs/``.

If any gate fails the command exits with an error and **nothing is published**: the previous
outputs stay in place.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

from plpredictor import config
from plpredictor.data import build_dataset
from plpredictor.data.sources import fixtures_api
from plpredictor.data.sources import football_data_uk as fduk

STATE_PATH = config.OUTPUTS_DIR / "refresh_state.json"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def changed(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Names of inputs whose hash differs from the saved state."""
    return sorted(k for k, v in new.items() if old.get(k) != v)


def gate_not_regressing(prev_meta: dict | None, played_now: int) -> list[str]:
    """Gate: the current season's played-match count must not drop below the last published one."""
    if prev_meta and played_now < prev_meta.get("matches_played", 0):
        return [f"played matches fell from {prev_meta['matches_played']} to {played_now}"]
    return []


def refresh(force: bool = False) -> dict:
    """Run the refresh. Returns a small report dict; raises if a validation gate fails."""
    today = dt.date.today().isoformat()
    cur_start = config.current_season_start()
    divs = (config.LEAGUE, config.SECOND_TIER)

    # 1. downloads (history is fetched once if missing; the current season is always re-checked)
    downloads: dict[str, bytes] = {}
    for div in divs:
        for start in config.season_starts():
            if start == cur_start:
                try:
                    downloads[f"{div}:{start}"] = fduk.fetch_bytes(start, div)
                except Exception as exc:  # e.g. new season file not published yet
                    print(f"  could not fetch current {div} file: {exc}")
            else:
                fduk.download_season(start, div)  # cached, only fetched when missing

    hashes = {k: sha256(v) for k, v in downloads.items()}
    fixtures_json = None
    try:
        fixtures_json = json.dumps(fixtures_api.fetch_upcoming(), sort_keys=True)
        hashes["fixtures"] = sha256(fixtures_json.encode())
    except fixtures_api.MissingKeyError:
        if fixtures_api.load_cached() is None:
            raise  # nothing cached either: publishing would lose every fixture date
        print("  no API key set: skipping the fixtures download (using cached fixtures)")

    # 2. change detection
    diff = changed(load_state().get("hashes", {}), hashes)
    if not diff and not force:
        print("Nothing changed since the last run. Stopping.")
        return {"changed": False}
    print("Changed inputs:", diff or "(forced)")

    # 3. snapshots + rebuild through the validation gates. On ANY failure, restore the old
    #    cache files so a bad download can never linger, and leave the published outputs untouched.
    backups = {k: fduk.load_cached(int(k.split(":")[1]), k.split(":")[0]) for k in downloads}
    try:
        for key, content in downloads.items():
            div, start = key.split(":")
            fduk.write_cache(int(start), div, content, snapshot_date=today)
        meta_path = config.OUTPUTS_DIR / "metadata.json"
        prev_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        merged = build_dataset.build()  # raises if validation fails or the download looks truncated
        season = config.season_label(cur_start)
        played_now = int(
            (
                (merged["div"] == config.LEAGUE)
                & (merged["season"] == season)
                & (merged["status"] == "played")
            ).sum()
        )
        problems = gate_not_regressing(prev_meta, played_now)
        if problems:
            raise ValueError(f"Refusing to publish: {problems}")

        # 4. predictions + simulation (only reached when every gate passed)
        from plpredictor.simulation import run as sim_run

        result = sim_run.run()
    except Exception:
        for key, old in backups.items():
            if old is not None:
                div, start = key.split(":")
                fduk.write_cache(int(start), div, old)
        raise

    STATE_PATH.write_text(json.dumps({"hashes": hashes, "updated": today}, indent=2))
    return {"changed": True, "data_as_of": result["meta"]["data_as_of"]}


def main(force: bool = False) -> None:
    report = refresh(force=force)
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if report['changed'] else 'false'}\n")
            f.write(f"data_as_of={report.get('data_as_of', '')}\n")
    print(report)
