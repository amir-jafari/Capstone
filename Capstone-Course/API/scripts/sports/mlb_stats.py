META = dict(
    name="MLB Stats API",
    domain="Sports",
    docs="https://github.com/toddrob99/MLB-StatsAPI/wiki (community docs) · terms: http://gdx.mlb.com/components/copyright.txt",
    auth="None",
    signup="No registration. Public but not formally documented; MLB's copyright notice limits use to individual, "
           "non-commercial, non-bulk purposes. Python wrapper: `pip install MLB-StatsAPI`.",
    provides=["Schedules, live game feeds with pitch-by-pitch data (velocity, spin, location)",
              "Standings, rosters, player season & career stats, team stats", "Minor-league and historical data"],
    ideas="Pitch-outcome prediction, player-valuation models, in-game win probability.",
    endpoint="GET https://statsapi.mlb.com/api/v1/{resource}",
    params=[
        ("standings?leagueId=&season=", "103 = American League, 104 = National League", "leagueId=103,104&season=2026"),
        ("schedule?sportId=&date= / startDate=&endDate=", "sportId 1 = MLB; dates MM/DD/YYYY or YYYY-MM-DD", "sportId=1&date=2026-09-28"),
        ("game/{gamePk}/feed/live (v1.1)", "Full live feed incl. every pitch (/api/v1.1/game/{gamePk}/feed/live)", "gamePk=822679 (from /schedule)"),
        ("people/{id}/stats?stats=&group=&season=", "Player stats; stats season | career | gameLog, group hitting | pitching", "stats=season&group=hitting"),
        ("teams/{id}/roster", "Active roster (rosterType=40Man for the 40-man roster)", "teams/120/roster"),
        ("hydrate", "Embed related objects in one call", "hydrate=team,venue"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://statsapi.mlb.com/api/v1/standings", params={"leagueId": "103,104", "season": 2026})
    rows = [t for rec in d["records"] for t in rec["teamRecords"]]
    rows.sort(key=lambda t: -float(t["winningPercentage"]))
    show([{"team": t["team"]["name"], "W": t["wins"], "L": t["losses"], "pct": t["winningPercentage"],
           "run_diff": t["runDifferential"]} for t in rows], max_items=6)


if __name__ == "__main__":
    main()
