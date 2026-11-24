META = dict(
    name="Spaceflight News API (SNAPI)",
    domain="Science & Research",
    docs="https://api.spaceflightnewsapi.net/v4/docs/",
    auth="None",
    signup="No registration. Aggregates space news from ~40 outlets (NASA, ESA, SpaceNews, Spaceflight Now, ...).",
    provides=["Articles, blogs and reports: title, summary, source site, publish date, image",
              "Links to related launches and events (Launch Library 2 IDs)"],
    ideas="Space-industry topic and sentiment tracking, linking news coverage to launch outcomes.",
    endpoint="GET https://api.spaceflightnewsapi.net/v4/{articles | blogs | reports}/",
    params=[
        ("search / title_contains", "Text search", "artemis"),
        ("news_site", "Filter by outlet(s), comma list", "NASA,SpaceNews"),
        ("published_at_gte / published_at_lte", "Date range", "2026-01-01"),
        ("has_launch", "true -> only articles linked to a launch", "true"),
        ("ordering", "-published_at | published_at", "-published_at"),
        ("limit / offset", "Paging", "5"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.spaceflightnewsapi.net/v4/articles/", params={
        "search": "artemis", "published_at_gte": "2026-01-01", "ordering": "-published_at", "limit": 100})
    show({"artemis_articles_2026": d["count"],
          "top_sources": Counter(a["news_site"] for a in d["results"]).most_common(4),
          "latest": [{"title": a["title"], "site": a["news_site"], "published": a["published_at"][:10]}
                     for a in d["results"][:3]]})


if __name__ == "__main__":
    main()
