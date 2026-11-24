META = dict(
    name="CollegeFootballData.com API",
    domain="Sports",
    docs="https://api.collegefootballdata.com/",
    auth="API key (Bearer token)",
    env="CFBD_API_KEY",
    signup="Get a free key at https://collegefootballdata.com/key (emailed). Free tier: 1,000 calls/month; Patreon "
           "tiers raise it. Python package: `pip install cfbd`.",
    provides=["Every FBS/FCS game since 1869: scores, venues, attendance, betting lines, Elo ratings",
              "Play-by-play with EPA, drives, team and player season stats, recruiting rankings, AP/coaches polls",
              "Advanced metrics: SP+, success rate, explosiveness, win probability"],
    ideas="Recruiting rankings vs. on-field results, conference realignment effects, upset prediction.",
    endpoint="GET https://api.collegefootballdata.com/{resource}",
    params=[
        ("games?year=&week=&team=&conference=", "Games and results", "year=2025&team=Navy"),
        ("rankings?year=&week=", "AP, Coaches and Playoff polls", "year=2025&week=10"),
        ("stats/season?year=&team=", "Team season stats", "—"),
        ("recruiting/teams?year=", "Team recruiting rankings", "year=2025"),
        ("plays?year=&week=&team=", "Play-by-play", "—"),
        ("Authorization (header)", "Bearer <key>", "Bearer <CFBD_API_KEY>"),
    ],
    sample="""[
  {"week": 1, "home": "Navy", "home_points": 52, "away": "VMI", "away_points": 7},
  {"week": 2, "home": "Navy", "home_points": 38, "away": "Temple", "away_points": 11}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    games = get("https://api.collegefootballdata.com/games", headers={"Authorization": f"Bearer {key('CFBD_API_KEY')}"},
                params={"year": 2025, "team": "Navy"})
    show([{"week": g["week"], "home": g["homeTeam"], "home_points": g["homePoints"], "away": g["awayTeam"],
           "away_points": g["awayPoints"]} for g in games[:6]])


if __name__ == "__main__":
    main()
