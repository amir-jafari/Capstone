META = dict(
    name="Wikimedia Feed API (most read, in the news, on this day)",
    domain="News & Media",
    docs="https://api.wikimedia.org/wiki/Feed_API",
    auth="None (anonymous: 500 requests/hour; free token for more)",
    signup="No registration for low volume. Create a free API token at https://api.wikimedia.org for 5,000 requests/hour.",
    provides=["Daily most-read Wikipedia articles with view counts (a real-time public-attention signal)",
              "'In the news' items with linked articles, today's featured article, picture of the day",
              "'On this day' historical events, births and deaths; available for many languages"],
    ideas="Daily attention dynamics around news events, comparing attention across languages, event detection.",
    endpoint="GET https://api.wikimedia.org/feed/v1/wikipedia/{lang}/{featured | onthisday}/...",
    params=[
        ("featured/{YYYY}/{MM}/{DD}", "Featured content for a day: tfa, mostread, news, image, onthisday", "featured/2026/10/11"),
        ("onthisday/{type}/{MM}/{DD}", "type = all | selected | births | deaths | events | holidays", "onthisday/events/10/12"),
        ("lang", "Wikipedia language code", "en, de, es, zh"),
        ("Authorization (header)", "Optional Bearer token", "—"),
    ],
)
import datetime as dt

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    day = dt.date.today() - dt.timedelta(days=1)
    d = get(f"https://api.wikimedia.org/feed/v1/wikipedia/en/featured/{day:%Y/%m/%d}")
    skip = ("Main_Page", "Special:")
    show({"date": day.isoformat(), "featured_article": d["tfa"]["titles"]["normalized"],
          "most_read": [{"title": a["titles"]["normalized"], "views": a["views"]}
                        for a in d["mostread"]["articles"] if not a["title"].startswith(skip)][:6],
          "in_the_news": [n["links"][0]["titles"]["normalized"] for n in d.get("news", [])][:4]})


if __name__ == "__main__":
    main()
