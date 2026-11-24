META = dict(
    name="TMDB (The Movie Database)",
    domain="Culture, Books & Entertainment",
    docs="https://developer.themoviedb.org/docs",
    auth="API read-access token (Bearer)",
    env="TMDB_TOKEN",
    signup="Create an account at https://www.themoviedb.org/signup -> Settings -> API -> request a Developer key "
           "(short form). Use the 'API Read Access Token' as a Bearer token. ~50 req/sec.",
    provides=["Movies & TV: title, release date, genres, budget, revenue, runtime, popularity, ratings",
              "Cast & crew credits, keywords, reviews, streaming providers by country"],
    ideas="Box-office prediction, recommender systems, gender representation in film crews.",
    sample="""[
  {"title": "Example Blockbuster", "release_date": "2026-10-03", "vote_average": 7.6, "vote_count": 812, "popularity": 1543.2}
]""",
    endpoint="GET https://api.themoviedb.org/3/trending/movie/{time_window}",
    params=[
        ("time_window", "day | week", "week"),
        ("/3/search/movie?query=&year=", "Sibling endpoint: search by title", "query=Inception"),
        ("/3/discover/movie", "Filter: with_genres, primary_release_year, vote_count.gte, sort_by (popularity.desc, revenue.desc, ...)", "sort_by=revenue.desc&primary_release_year=2025"),
        ("/3/movie/{id}?append_to_response=", "Details + credits, keywords, reviews in one call", "append_to_response=credits,keywords"),
        ("language / page", "Locale and paging", "en-US / 1"),
        ("Authorization (header)", "Read-access token", "Bearer <TMDB_TOKEN>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.themoviedb.org/3/trending/movie/week",
            headers={"Authorization": f"Bearer {key('TMDB_TOKEN')}"})
    show([{k: m[k] for k in ("title", "release_date", "vote_average", "vote_count", "popularity")}
          for m in d["results"]])


if __name__ == "__main__":
    main()
