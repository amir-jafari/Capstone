META = dict(
    name="Cricsheet (ball-by-ball cricket data)",
    domain="Sports",
    docs="https://cricsheet.org/format/json/",
    auth="None (zip downloads)",
    signup="No registration; no REST API. One JSON file per match, zipped by competition (IPL, Tests, ODIs, T20Is, "
           "women's cricket, ...). ODC-BY licence.",
    provides=["Every delivery: batter, bowler, runs, extras, wickets (kind, fielders), over/ball number",
              "Match info: teams, venue, dates, toss, result, player of the match, officials", "18,000+ matches since 2001"],
    ideas="Win-probability models for T20, toss-decision analysis, player-impact metrics, death-over strategies.",
    endpoint="GET https://cricsheet.org/downloads/{collection}_json.zip",
    params=[
        ("collection", "ipl | tests | odis | t20s | all | recently_added_7 | ...", "ipl"),
        ("file contents", "{match_id}.json with meta, info, innings[].overs[].deliveries[]", "—"),
        ("info fields", "dates, teams, venue, toss {winner, decision}, outcome {winner, by}, player_of_match", "—"),
        ("csv2 format", "{collection}_csv2.zip for one-row-per-ball CSVs", "ipl_csv2.zip"),
    ],
)
import io
import json
import zipfile
from collections import Counter

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show


def main():
    z = zipfile.ZipFile(io.BytesIO(requests.get("https://cricsheet.org/downloads/ipl_json.zip",
                                                headers={"User-Agent": USER_AGENT}, timeout=120).content))
    infos = [json.loads(z.read(n))["info"] for n in z.namelist() if n.endswith(".json")]
    season = max(str(i["season"]) for i in infos)
    latest = [i for i in infos if str(i["season"]) == season]
    decided = [i for i in latest if i["outcome"].get("winner")]  # excludes no-results and ties
    show({"ipl_matches": len(infos), "latest_season": season, "matches_in_season": len(latest),
          "most_wins": Counter(i["outcome"]["winner"] for i in decided).most_common(4),
          "toss_winner_won_match_pct": round(100 * sum(i["toss"]["winner"] == i["outcome"]["winner"] for i in decided) / len(decided)),
          "chose_to_field_after_toss_pct": round(100 * sum(i["toss"]["decision"] == "field" for i in latest) / len(latest))})


if __name__ == "__main__":
    main()
