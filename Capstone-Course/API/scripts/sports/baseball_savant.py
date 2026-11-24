META = dict(
    name="Baseball Savant (MLB Statcast)",
    domain="Sports",
    docs="https://baseballsavant.mlb.com/csv-docs",
    auth="None (CSV endpoints)",
    signup="No registration; unofficial CSV endpoints behind MLB's Statcast site. The pitch-level search returns at most "
           "~25,000 rows per request, so query a few days at a time. Python package: `pip install pybaseball`.",
    provides=["Pitch-level Statcast data: velocity, spin rate, movement, release point, exit velocity, launch angle, location",
              "Leaderboards: expected stats (xBA, xSLG, xwOBA), sprint speed, outs above average, bat tracking, arsenal",
              "Player and game identifiers matching the MLB Stats API"],
    ideas="Luck vs. skill (actual vs. expected stats), pitch-outcome models, swing/launch-angle optimization.",
    endpoint="GET https://baseballsavant.mlb.com/{leaderboard/{name} | statcast_search/csv}",
    params=[
        ("leaderboard/expected_statistics?type=&year=&min=&csv=true", "xBA/xSLG/xwOBA vs. actual; type batter | pitcher; min=q qualified", "type=batter&year=2025&min=q"),
        ("statcast_search/csv?all=true&type=details", "Pitch-level rows", "—"),
        ("game_date_gt / game_date_lt", "Date range for pitch search (keep it short)", "2025-09-01 / 2025-09-02"),
        ("hfSea / hfGT / player_type", "Season (2025|), game type (R| regular season), batter | pitcher", "hfSea=2025%7C"),
        ("other leaderboards", "sprint_speed, outs_above_average, bat-tracking, pitch-arsenal-stats (add csv=true)", "—"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    text = get("https://baseballsavant.mlb.com/leaderboard/expected_statistics",
               params={"type": "batter", "year": 2025, "min": "q", "csv": "true"}, as_json=False)
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    luck = sorted(rows, key=lambda r: float(r["est_woba_minus_woba_diff"]))
    fmt = lambda r: {"player": r["last_name, first_name"], "woba": float(r["woba"]), "xwoba": float(r["est_woba"])}
    show({"qualified_batters_2025": len(rows),
          "top_xwOBA": [fmt(r) for r in sorted(rows, key=lambda r: -float(r["est_woba"]))[:3]],
          "most_unlucky (xwOBA > wOBA)": [fmt(r) for r in luck[:3]],
          "most_lucky (wOBA > xwOBA)": [fmt(r) for r in luck[-3:]]})


if __name__ == "__main__":
    main()
