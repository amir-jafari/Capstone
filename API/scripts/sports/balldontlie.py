META = dict(
    name="BALLDONTLIE (NBA, NFL, MLB, ...)",
    domain="Sports",
    docs="https://docs.balldontlie.io/",
    auth="API key (header `Authorization`)",
    env="BALLDONTLIE_API_KEY",
    signup="Sign up at https://app.balldontlie.io -> key on the dashboard. Free tier: teams, players and games "
           "(5 req/min); paid tiers add box scores, season averages, odds and more sports.",
    provides=["NBA teams, players, games back to 1946 (free)", "Box scores, player stats, season averages, injuries, betting odds (paid)",
              "Same API shape for NFL, MLB, NHL, EPL, WNBA, NCAA"],
    ideas="Player-performance forecasting, schedule-fatigue effects, team-strength ratings.",
    endpoint="GET https://api.balldontlie.io/v1/{resource}",
    params=[
        ("resource", "teams | players | games | stats | season_averages | box_scores (some paid)", "games"),
        ("seasons[]", "Season start year (repeatable)", "2025"),
        ("team_ids[] / player_ids[]", "Filter by IDs (repeatable)", "14"),
        ("dates[] / start_date / end_date", "YYYY-MM-DD", "2026-01-15"),
        ("search", "Player name search (players)", "curry"),
        ("per_page / cursor", "Max 100; cursor-based paging (meta.next_cursor)", "25"),
        ("Authorization (header)", "Your API key", "<BALLDONTLIE_API_KEY>"),
    ],
    sample="""[
  {"date": "2026-04-12", "home": "Boston Celtics", "home_score": 112, "visitor": "Washington Wizards", "visitor_score": 98, "status": "Final"},
  {"date": "2026-04-10", "home": "Washington Wizards", "home_score": 104, "visitor": "Miami Heat", "visitor_score": 109, "status": "Final"}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.balldontlie.io/v1/games", headers={"Authorization": key("BALLDONTLIE_API_KEY")},
            params={"seasons[]": 2025, "team_ids[]": 30, "per_page": 5})  # 30 = Washington Wizards
    show([{"date": g["date"], "home": g["home_team"]["full_name"], "home_score": g["home_team_score"],
           "visitor": g["visitor_team"]["full_name"], "visitor_score": g["visitor_team_score"], "status": g["status"]}
          for g in d["data"]])


if __name__ == "__main__":
    main()
