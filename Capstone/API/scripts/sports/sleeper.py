META = dict(
    name="Sleeper API (fantasy football)",
    domain="Sports",
    docs="https://docs.sleeper.com/",
    auth="None",
    signup="No registration; read-only. Stay under 1,000 calls/minute. Download the full player list (/players/nfl, "
           "~5 MB) at most once a day and cache it.",
    provides=["Trending players (most added/dropped in fantasy leagues over the last N hours)",
              "All NFL players with team, position, injury status, depth chart, IDs for other sites",
              "Public leagues, rosters, matchups, drafts; current NFL week/season state"],
    ideas="Fantasy-market sentiment (adds/drops) vs. performance, injury news effects, draft-strategy analysis.",
    endpoint="GET https://api.sleeper.app/v1/{resource}",
    params=[
        ("players/nfl/trending/{add | drop}?lookback_hours=&limit=", "Trending players", "add?lookback_hours=24&limit=10"),
        ("players/nfl", "Every player keyed by player_id (large; cache daily)", "—"),
        ("state/nfl", "Current season, week and season type", "—"),
        ("league/{league_id} · /rosters · /matchups/{week}", "A public league", "—"),
        ("draft/{draft_id}/picks", "Draft results", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

API = "https://api.sleeper.app/v1"


def main():
    state = get(f"{API}/state/nfl")
    adds = get(f"{API}/players/nfl/trending/add", params={"lookback_hours": 24, "limit": 6})
    players = get(f"{API}/players/nfl", timeout=120)  # ~5 MB: cache this in real projects
    show({"season": state["season"], "week": state["week"], "most_added_last_24h": [
        {"player": players.get(a["player_id"], {}).get("full_name"), "team": players.get(a["player_id"], {}).get("team"),
         "pos": players.get(a["player_id"], {}).get("position"),
         "injury": players.get(a["player_id"], {}).get("injury_status"), "adds": a["count"]} for a in adds]})


if __name__ == "__main__":
    main()
