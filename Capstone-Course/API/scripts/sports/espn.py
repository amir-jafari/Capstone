META = dict(
    name="ESPN Site API (unofficial)",
    domain="Sports",
    docs="https://github.com/pseudo-r/Public-ESPN-API (community docs; no official documentation)",
    auth="None",
    signup="No registration. Unofficial and undocumented: endpoints can change without notice and ESPN's terms "
           "apply. It rejects some custom User-Agents, so this example sends the default `requests` one.",
    provides=["Scoreboards with live/final scores for NFL, NBA, MLB, NHL, college sports, soccer, ...",
              "Teams, rosters, schedules, standings, news headlines", "Game summaries: box scores, play-by-play, win probability"],
    ideas="Win-probability models, home-advantage analysis, sports-news NLP.",
    endpoint="GET https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/{resource}",
    params=[
        ("sport / league", "football/nfl, basketball/nba, baseball/mlb, hockey/nhl, football/college-football, soccer/eng.1, ...", "football/nfl"),
        ("resource", "scoreboard | teams | teams/{id} | teams/{id}/roster | teams/{id}/schedule | news | summary", "scoreboard"),
        ("dates", "Scoreboard date or range, YYYYMMDD or YYYYMMDD-YYYYMMDD", "20261011"),
        ("week / seasontype", "NFL/college week number; seasontype 1 = pre, 2 = regular, 3 = post", "6 / 2"),
        ("summary?event=", "Box score + play-by-play for one game (event id from the scoreboard)", "event=401872993"),
        ("limit", "Max events/items", "100"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            headers={"User-Agent": requests.utils.default_user_agent()})
    show({"season": d["season"]["year"], "week": d.get("week", {}).get("number"), "games": [
        {"name": e["shortName"], "status": e["status"]["type"]["description"],
         "score": " - ".join(f'{c["team"]["abbreviation"]} {c.get("score")}' for c in e["competitions"][0]["competitors"])}
        for e in d["events"]]})


if __name__ == "__main__":
    main()
