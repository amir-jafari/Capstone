META = dict(
    name="nflverse data (NFL play-by-play & stats)",
    domain="Sports",
    docs="https://github.com/nflverse/nflverse-data",
    auth="None (CSV / Parquet downloads from GitHub releases)",
    signup="No registration; no REST API. Files are GitHub release assets (CSV, Parquet, RDS), updated nightly in season. "
           "Python package: `pip install nflreadpy`. CC BY 4.0.",
    provides=["Play-by-play since 1999 with EPA, win probability, CPOE, air yards", "Weekly and season player and team stats",
              "Schedules with results, betting lines and weather; rosters, injuries, depth charts, draft picks, contracts, Next Gen Stats"],
    ideas="Win-probability and 4th-down decision models, player-value metrics (EPA), betting-line efficiency.",
    endpoint="GET https://github.com/nflverse/nflverse-data/releases/download/{release}/{file}",
    params=[
        ("schedules/games.csv", "Every game since 1999: scores, spread_line, total_line, roof, temp, wind", "—"),
        ("stats_player/stats_player_week_{season}.csv", "Weekly player stats (the old player_stats release is legacy)", "stats_player_week_2026.csv"),
        ("pbp/play_by_play_{season}.parquet", "Play-by-play (~40 MB per season as CSV; use Parquet)", "play_by_play_2025.parquet"),
        ("rosters / injuries / draft_picks / contracts / nextgen_stats", "Other releases (list via the GitHub releases page)", "—"),
        ("format", ".csv | .csv.gz | .parquet | .rds", ".parquet"),
    ],
)
import csv
import io
from collections import defaultdict

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def main():
    rows = list(csv.DictReader(io.StringIO(get(f"{BASE}/stats_player/stats_player_week_2026.csv", as_json=False))))
    totals = defaultdict(lambda: {"passing_yards": 0, "rushing_yards": 0, "receiving_yards": 0, "team": ""})
    for r in rows:
        t = totals[r["player_display_name"]]
        t["team"] = r["team"]
        for k in ("passing_yards", "rushing_yards", "receiving_yards"):
            t[k] += float(r[k] or 0)
    top = lambda k: [{"player": p, "team": v["team"], k: int(v[k])} for p, v in sorted(totals.items(), key=lambda kv: -kv[1][k])[:3]]
    show({"season": 2026, "weeks_in_file": len({r["week"] for r in rows}),
          "passing_leaders": top("passing_yards"), "rushing_leaders": top("rushing_yards"), "receiving_leaders": top("receiving_yards")})


if __name__ == "__main__":
    main()
