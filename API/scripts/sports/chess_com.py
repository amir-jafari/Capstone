META = dict(
    name="Chess.com Published-Data API",
    domain="Sports",
    docs="https://www.chess.com/news/view/published-data-api",
    auth="None",
    signup="No registration. Read-only public data; serial requests are unlimited, parallel requests may be rate-limited "
           "(HTTP 429). Complements the Lichess entry.",
    provides=["Leaderboards for daily, rapid, blitz, bullet, puzzles", "Player profiles and ratings; titled players (GM, IM, WGM...)",
              "Monthly game archives with full PGN (moves, clocks, openings)", "Clubs, tournaments, team matches, country players"],
    ideas="Rating-system analysis, opening popularity by rating, cheating detection, time-pressure blunders.",
    endpoint="GET https://api.chess.com/pub/{resource}",
    params=[
        ("leaderboards", "All leaderboards", "—"),
        ("titled/{title}", "Usernames with a title: GM | WGM | IM | WIM | FM | ...", "titled/GM"),
        ("player/{username} · player/{username}/stats", "Profile and ratings", "player/hikaru/stats"),
        ("player/{username}/games/archives · /games/{YYYY}/{MM}", "Monthly game archives (PGN)", "player/hikaru/games/2026/09"),
        ("country/{iso}/players", "Players by country", "country/US/players"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    lb = get("https://api.chess.com/pub/leaderboards")
    gms = get("https://api.chess.com/pub/titled/GM")["players"]
    stats = get("https://api.chess.com/pub/player/hikaru/stats")
    show({"titled_GMs_on_chess_com": len(gms),
          "live_blitz_top5": [{"rank": p["rank"], "username": p["username"], "title": p.get("title"), "rating": p["score"]}
                              for p in lb["live_blitz"][:5]],
          "hikaru_ratings": {k.replace("chess_", ""): v["last"]["rating"] for k, v in stats.items() if k.startswith("chess_")}})


if __name__ == "__main__":
    main()
