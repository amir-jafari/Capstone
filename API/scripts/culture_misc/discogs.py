META = dict(
    name="Discogs API (music releases & marketplace)",
    domain="Culture, Books & Entertainment",
    docs="https://www.discogs.com/developers",
    auth="None for public data (token raises limits and unlocks images)",
    env="DISCOGS_TOKEN (optional)",
    signup="Public data works without auth at 25 requests/minute; a personal token from "
           "https://www.discogs.com/settings/developers gives 60/minute. A descriptive User-Agent is required.",
    provides=["16M+ releases (vinyl, CD, cassette...) with labels, formats, countries, credits, tracklists",
              "Artists and their discographies; master releases grouping editions",
              "Community data: how many users have/want a release; marketplace price statistics"],
    ideas="Vinyl revival analysis, collectibility (want/have ratios) vs. price, label and credit networks.",
    endpoint="GET https://api.discogs.com/{resource}",
    params=[
        ("database/search?q=&type=", "Search; type release | master | artist | label; filters genre, style, year, format, country", "q=nirvana&type=master"),
        ("artists/{id}/releases", "An artist's discography (sort=year, sort_order)", "artists/125246/releases?sort=year"),
        ("releases/{id} · masters/{id}", "Release details incl. community have/want", "releases/249504"),
        ("marketplace/stats/{release_id}", "Lowest price and number for sale", "—"),
        ("per_page / page", "Max 100", "5"),
        ("Authorization (header)", "Discogs token=<token> (optional)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.discogs.com/artists/125246/releases", params={"sort": "year", "sort_order": "asc",
                                                                      "per_page": 100})  # Nirvana
    masters = [r for r in d["releases"] if r["type"] == "master" and r.get("role") == "Main"]
    rel = get("https://api.discogs.com/masters/13814")  # Nevermind
    show({"artist": "Nirvana", "discography_entries": d["pagination"]["items"],
          "main_masters": [f'{r.get("year")} {r["title"]}' for r in masters][:6],
          "nevermind": {"year": rel["year"], "genres": rel["genres"], "styles": rel["styles"],
                        "tracks": len(rel["tracklist"])}})


if __name__ == "__main__":
    main()
