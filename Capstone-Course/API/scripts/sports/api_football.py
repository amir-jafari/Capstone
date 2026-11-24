META = dict(
    name="API-Football (API-Sports)",
    domain="Sports",
    docs="https://www.api-football.com/documentation-v3",
    auth="API key (header `x-apisports-key`)",
    env="APISPORTS_KEY",
    signup="Register at https://dashboard.api-football.com/register for a free key: 100 requests/day. The free plan only "
           "covers some past seasons - check the dashboard. The same account works for API-Sports' basketball, NFL, "
           "hockey, F1 and other APIs.",
    provides=["1,100+ soccer leagues and cups: fixtures, live scores, standings, lineups, events, statistics",
              "Players and squads with season stats, top scorers, transfers, injuries", "Pre-match odds and predictions"],
    ideas="Match-outcome prediction with lineups and stats, transfer-market analysis, injury impact on results.",
    endpoint="GET https://v3.football.api-sports.io/{resource}",
    params=[
        ("leagues?country= / ?search=", "Find league IDs (39 Premier League, 140 La Liga, 2 Champions League)", "country=England"),
        ("standings?league=&season=", "League table", "league=39&season=2023"),
        ("fixtures?league=&season= / ?date= / ?live=all", "Fixtures and results", "date=2026-10-12"),
        ("players/topscorers?league=&season=", "Top scorers", "league=39&season=2023"),
        ("x-apisports-key (header)", "Your key", "<APISPORTS_KEY>"),
    ],
    sample="""[
  {"rank": 1, "team": "Manchester City", "played": 38, "points": 91, "goal_diff": 62},
  {"rank": 2, "team": "Arsenal", "played": 38, "points": 89, "goal_diff": 62}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://v3.football.api-sports.io/standings", headers={"x-apisports-key": key("APISPORTS_KEY")},
            params={"league": 39, "season": 2023})
    table = d["response"][0]["league"]["standings"][0]
    show([{"rank": t["rank"], "team": t["team"]["name"], "played": t["all"]["played"], "points": t["points"],
           "goal_diff": t["goalsDiff"]} for t in table[:5]])


if __name__ == "__main__":
    main()
