META = dict(
    name="TheSportsDB",
    domain="Sports",
    docs="https://www.thesportsdb.com/documentation",
    auth="API key in the URL path (free test key `123`)",
    env="THESPORTSDB_KEY (optional)",
    signup="The free key `123` works for v1 lookups (~30 req/min). A paid Patreon key unlocks v2, livescores and higher limits.",
    provides=["Teams, players, leagues across 50+ sports (soccer, NFL, NBA, cricket, rugby, motorsport, esports...)",
              "Past & upcoming events with scores, venues, stadium/badge/kit artwork", "League tables for major soccer leagues"],
    ideas="Multi-sport metadata joins, image datasets (badges/kits), schedule analysis.",
    endpoint="GET https://www.thesportsdb.com/api/v1/json/{key}/{endpoint}.php",
    params=[
        ("key", "123 (free) or your premium key", "123"),
        ("searchteams.php?t=", "Search teams by name", "t=Arsenal"),
        ("searchplayers.php?p=", "Search players by name", "p=Bukayo Saka"),
        ("lookuptable.php?l=&s=", "League table (league id + season)", "l=4328&s=2025-2026"),
        ("eventsnextleague.php?id= / eventspastleague.php?id=", "Upcoming / recent events in a league", "id=4328"),
        ("eventsseason.php?id=&s=", "All events in a league season", "id=4328&s=2025-2026"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    k = os.getenv("THESPORTSDB_KEY", "123")
    team = get(f"https://www.thesportsdb.com/api/v1/json/{k}/searchteams.php", params={"t": "Arsenal"})["teams"][0]
    past = get(f"https://www.thesportsdb.com/api/v1/json/{k}/eventslast.php", params={"id": team["idTeam"]})
    show({"team": {k2: team[k2] for k2 in ("strTeam", "strLeague", "intFormedYear", "strStadium")},
          "last_results": [{k2: e[k2] for k2 in ("dateEvent", "strEvent", "intHomeScore", "intAwayScore")}
                           for e in (past.get("results") or [])]})


if __name__ == "__main__":
    main()
