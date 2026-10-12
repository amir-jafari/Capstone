META = dict(
    name="NHL Web API (unofficial)",
    domain="Sports",
    docs="https://github.com/Zmalski/NHL-API-Reference (community docs)",
    auth="None",
    signup="No registration. Undocumented public API behind nhl.com; endpoints can change.",
    provides=["Standings, schedules, scores", "Play-by-play with shot coordinates and event types",
              "Player game logs and season stats, team rosters"],
    ideas="Expected-goals (xG) models from shot locations, goalie performance analysis.",
    endpoint="GET https://api-web.nhle.com/v1/{resource}",
    params=[
        ("standings/{date | now}", "League standings on a date (YYYY-MM-DD)", "standings/now"),
        ("schedule/{date | now}", "Week of games starting at the date", "schedule/2026-10-11"),
        ("score/{date | now}", "Scores for a date", "score/now"),
        ("gamecenter/{gameId}/play-by-play", "Every event with x/y coordinates", "gamecenter/2026020001/play-by-play"),
        ("roster/{team}/current", "Team roster (3-letter team code)", "roster/WSH/current"),
        ("player/{playerId}/landing", "Player bio + career stats", "player/8471214/landing"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api-web.nhle.com/v1/standings/now")
    rows = sorted(d["standings"], key=lambda t: t["leagueSequence"])
    show({"as_of": d["standingsDateTimeUtc"], "top_teams": [
        {"team": t["teamName"]["default"], "GP": t["gamesPlayed"], "W": t["wins"], "L": t["losses"],
         "PTS": t["points"], "GF": t["goalFor"], "GA": t["goalAgainst"]} for t in rows]})


if __name__ == "__main__":
    main()
