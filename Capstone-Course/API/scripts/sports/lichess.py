META = dict(
    name="Lichess API (chess)",
    domain="Sports",
    docs="https://lichess.org/api",
    auth="None for public data (OAuth token for account actions)",
    signup="No registration for public endpoints; one request at a time, back off on HTTP 429. "
           "The full game database (billions of games, PGN) is free at https://database.lichess.org.",
    provides=["Leaderboards by time control, user profiles & rating histories",
              "Game export as PGN/NDJSON with moves, clocks and engine evals", "Opening explorer statistics (token required), puzzles"],
    ideas="Rating-dynamics modeling, opening-success analysis, cheating/anomaly detection.",
    endpoint="GET https://lichess.org/api/{resource}",
    params=[
        ("player/top/{nb}/{perfType}", "Leaderboard; nb ≤ 200; perfType bullet | blitz | rapid | classical | ...", "player/top/10/blitz"),
        ("user/{username}", "Profile with ratings per time control", "user/DrNykterstein"),
        ("user/{username}/rating-history", "Rating over time", "user/DrNykterstein/rating-history"),
        ("games/user/{username}?max=&perfType=&evals=&clocks=", "Export games (Accept: application/x-ndjson for JSON lines)", "max=10&perfType=blitz"),
        ("explorer.lichess.ovh/lichess?fen=&speeds=&ratings=", "Opening explorer statistics for a position (now requires a Lichess OAuth token)", "speeds=blitz&ratings=2000"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://lichess.org/api/player/top/5/blitz", headers={"Accept": "application/vnd.lichess.v3+json"})
    show([{"username": u["username"], "title": u.get("title"), "blitz_rating": u["perfs"]["blitz"]["rating"],
           "progress": u["perfs"]["blitz"]["progress"]} for u in d["users"]])


if __name__ == "__main__":
    main()
