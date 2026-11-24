META = dict(
    name="TVmaze API (TV shows & schedules)",
    domain="Culture, Books & Entertainment",
    docs="https://www.tvmaze.com/api",
    auth="None",
    signup="No registration. At least 20 calls / 10 seconds. CC BY-SA licence (attribute TVmaze).",
    provides=["TV shows: genres, network/streaming service, premiere/end dates, runtime, rating, status, summary",
              "Episodes with air dates, cast and crew, full daily schedules by country (incl. streaming 'web' schedule)",
              "Lookup by IMDb/TheTVDB IDs; changes feed for keeping a local copy up to date"],
    ideas="Streaming vs. broadcast release strategies, show cancellation prediction, ratings by genre.",
    endpoint="GET https://api.tvmaze.com/{resource}",
    params=[
        ("search/shows?q=", "Fuzzy show search with relevance score", "q=office"),
        ("shows/{id}?embed[]=", "Show details; embed episodes, cast, seasons", "shows/526?embed[]=cast"),
        ("schedule?country=&date=", "Everything airing on broadcast TV that day", "country=US&date=2026-10-12"),
        ("schedule/web?date=", "Streaming releases (Netflix, Prime, ...)", "date=2026-10-12"),
        ("lookup/shows?imdb=", "Find a show by IMDb ID (answers with a 301 redirect to the show record)", "imdb=tt0386676"),
        ("shows?page=", "Full index, 250 shows per page", "0"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    web = get("https://api.tvmaze.com/schedule/web", params={"date": "2026-10-10"})
    shows = {e["_embedded"]["show"]["id"]: e["_embedded"]["show"] for e in web if e.get("_embedded")}
    services = Counter((s.get("webChannel") or {}).get("name") for s in shows.values())
    show({"date": "2026-10-10", "streaming_episodes_released": len(web), "distinct_shows": len(shows),
          "by_service": services.most_common(5),
          "examples": [{"show": s["name"], "service": (s.get("webChannel") or {}).get("name"), "genres": s["genres"],
                        "rating": s["rating"]["average"]} for s in list(shows.values())[:3]]})


if __name__ == "__main__":
    main()
