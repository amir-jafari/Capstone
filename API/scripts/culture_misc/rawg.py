META = dict(
    name="RAWG Video Games Database API",
    domain="Culture, Books & Entertainment",
    docs="https://api.rawg.io/docs/",
    auth="API key (query `key`)",
    env="RAWG_API_KEY",
    signup="Get a free key at https://rawg.io/apidocs (20,000 requests/month on the free tier). Attribution to RAWG "
           "with a link is required.",
    provides=["500k+ games across all platforms: release date, genres, platforms, ESRB rating, Metacritic score, playtime",
              "User ratings distribution, tags, developers, publishers, stores, screenshots, achievements"],
    ideas="What predicts high ratings, platform exclusivity trends, release-calendar analysis.",
    endpoint="GET https://api.rawg.io/api/{games | games/{id} | genres | platforms | developers}",
    params=[
        ("search", "Text search", "zelda"),
        ("dates", "Release-date range", "2025-01-01,2025-12-31"),
        ("genres / platforms / tags", "IDs or slugs (comma list)", "genres=indie"),
        ("ordering", "name | released | added | created | rating | metacritic (prefix - to reverse)", "-metacritic"),
        ("metacritic", "Score range", "80,100"),
        ("page_size / page", "Max 40", "5"),
        ("key", "Your key", "<RAWG_API_KEY>"),
    ],
    sample="""[
  {"name": "Example Game", "released": "2025-05-12", "metacritic": 94, "rating": 4.6, "genres": ["Action", "Adventure"]}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.rawg.io/api/games", params={"key": key("RAWG_API_KEY"), "dates": "2025-01-01,2025-12-31",
                                                      "ordering": "-metacritic", "page_size": 5})
    show([{"name": g["name"], "released": g["released"], "metacritic": g["metacritic"], "rating": g["rating"],
           "genres": [x["name"] for x in g["genres"]]} for g in d["results"]])


if __name__ == "__main__":
    main()
