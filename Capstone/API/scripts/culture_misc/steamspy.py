META = dict(
    name="SteamSpy (Steam game statistics)",
    domain="Culture, Books & Entertainment",
    docs="https://steamspy.com/api.php",
    auth="None",
    signup="No registration. 1 request/second (the 'all' request: 1 per minute). Unofficial: ownership figures are "
           "estimated ranges, not exact counts. For official store data see the Steam Web API (key).",
    provides=["Per-game: estimated owners range, positive/negative reviews, average and median playtime, price, discounts, CCU",
              "Top 100 by recent players or all time; games by genre or tag; developer and publisher",
              "User-applied tags with vote counts and supported languages (appdetails)"],
    ideas="Game-success prediction from tags/price/reviews, playtime distributions, pricing and discount strategies.",
    endpoint="GET https://steamspy.com/api.php?request={request}",
    params=[
        ("request", "appdetails | top100in2weeks | top100forever | top100owned | genre | tag | all", "top100in2weeks"),
        ("appid", "For appdetails", "570 (Dota 2)"),
        ("genre / tag", "For genre / tag requests", "genre=Indie"),
        ("page", "For 'all' (1,000 games per page)", "0"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    top = get("https://steamspy.com/api.php", params={"request": "top100in2weeks"})
    games = sorted(top.values(), key=lambda g: -g["ccu"])[:5]
    show({"top_by_peak_concurrent_players_2_weeks": [
        {"name": g["name"], "developer": g["developer"], "owners": g["owners"], "peak_ccu": g["ccu"],
         "positive_review_pct": round(100 * g["positive"] / max(g["positive"] + g["negative"], 1)),
         "price_usd": int(g["price"] or 0) / 100} for g in games]})


if __name__ == "__main__":
    main()
