META = dict(
    name="Fantasy Premier League API",
    domain="Sports",
    docs="https://fantasy.premierleague.com/ (unofficial endpoints; community guide: https://github.com/vaastav/Fantasy-Premier-League)",
    auth="None",
    signup="No registration for public data; undocumented endpoints behind the official game, so be gentle and cache.",
    provides=["Every Premier League player: price, total points, form, minutes, goals, assists, expected goals/assists (xG/xA), ownership %",
              "Teams with strength ratings, gameweek deadlines, fixtures with difficulty ratings",
              "Player history by gameweek; public manager teams and leagues"],
    ideas="Points prediction and squad optimization (integer programming), price-change prediction, xG vs. actual goals.",
    endpoint="GET https://fantasy.premierleague.com/api/{resource}/",
    params=[
        ("bootstrap-static/", "Players (elements), teams, gameweeks (events), positions", "—"),
        ("fixtures/?event=", "Fixtures, optionally for one gameweek", "event=8"),
        ("element-summary/{player_id}/", "A player's gameweek history and upcoming fixtures", "element-summary/328/"),
        ("event/{gw}/live/", "Live points for every player in a gameweek", "—"),
        ("entry/{manager_id}/", "A public manager's team summary", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def main():
    d = get("https://fantasy.premierleague.com/api/bootstrap-static/")
    teams = {t["id"]: t["short_name"] for t in d["teams"]}
    gw = next((e for e in d["events"] if e["is_current"]), None)
    top = sorted(d["elements"], key=lambda p: -p["total_points"])[:6]
    show({"players": len(d["elements"]), "current_gameweek": gw and gw["name"], "top_scorers_in_fpl": [
        {"player": p["web_name"], "team": teams[p["team"]], "pos": POS[p["element_type"]], "price_m": p["now_cost"] / 10,
         "points": p["total_points"], "goals": p["goals_scored"], "xG": p.get("expected_goals"),
         "owned_pct": p["selected_by_percent"]} for p in top]})


if __name__ == "__main__":
    main()
