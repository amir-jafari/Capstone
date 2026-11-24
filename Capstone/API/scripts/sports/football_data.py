META = dict(
    name="football-data.org (soccer)",
    domain="Sports",
    docs="https://docs.football-data.org/general/v4/index.html",
    auth="API token (header `X-Auth-Token`)",
    env="FOOTBALL_DATA_TOKEN",
    signup="Register at https://www.football-data.org/client/register (token emailed). Free tier: 12 top competitions "
           "(Premier League, La Liga, Bundesliga, Serie A, Champions League, World Cup...), 10 req/min.",
    provides=["Competitions, standings (total/home/away), matches with scores and status",
              "Teams with squads, top scorers, head-to-head"],
    ideas="Match-outcome prediction (Poisson / Elo), home-advantage studies, league-table simulation.",
    endpoint="GET https://api.football-data.org/v4/{resource}",
    params=[
        ("competitions/{code}/standings", "League table; codes PL, PD, BL1, SA, FL1, CL, WC, ...", "competitions/PL/standings"),
        ("competitions/{code}/matches", "Fixtures/results; filter with matchday, status, dateFrom, dateTo, season", "matchday=7"),
        ("competitions/{code}/scorers", "Top scorers (limit=)", "limit=10"),
        ("teams/{id}/matches", "A team's matches (status, venue HOME | AWAY, limit)", "teams/57/matches?status=FINISHED"),
        ("season", "Season start year", "2025"),
        ("X-Auth-Token (header)", "Your token", "<FOOTBALL_DATA_TOKEN>"),
    ],
    sample="""[
  {"pos": 1, "team": "Arsenal FC", "played": 7, "won": 5, "draw": 1, "lost": 1, "points": 16, "goal_diff": 10},
  {"pos": 2, "team": "Liverpool FC", "played": 7, "won": 5, "draw": 0, "lost": 2, "points": 15, "goal_diff": 6}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.football-data.org/v4/competitions/PL/standings",
            headers={"X-Auth-Token": key("FOOTBALL_DATA_TOKEN")})
    table = d["standings"][0]["table"]
    show([{"pos": r["position"], "team": r["team"]["name"], "played": r["playedGames"], "won": r["won"],
           "draw": r["draw"], "lost": r["lost"], "points": r["points"], "goal_diff": r["goalDifference"]} for r in table])


if __name__ == "__main__":
    main()
