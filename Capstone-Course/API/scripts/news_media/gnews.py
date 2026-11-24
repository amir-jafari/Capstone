META = dict(
    name="GNews API",
    domain="News & Media",
    docs="https://gnews.io/docs/v4",
    auth="API key (query `apikey`)",
    env="GNEWS_API_KEY",
    signup="Sign up at https://gnews.io for a free key: 100 requests/day, up to 10 articles per request, development "
           "use. An alternative to NewsAPI.org with search by language and country.",
    provides=["Search articles from 80,000+ sources: title, description, content snippet, URL, image, published time, source",
              "Top headlines by category (world, nation, business, technology, science, health, sports, entertainment)",
              "Filters for language, country, date range, and which fields to search"],
    ideas="Multilingual news monitoring, headline classification, comparing coverage across countries.",
    endpoint="GET https://gnews.io/api/v4/{search | top-headlines}",
    params=[
        ("q", "Query with AND/OR/NOT and quotes", "\"artificial intelligence\" AND regulation"),
        ("lang / country", "Two-letter codes", "en / us"),
        ("from / to", "ISO 8601 dates", "2026-10-01T00:00:00Z"),
        ("in", "Fields to search: title, description, content", "title"),
        ("category (top-headlines)", "general | world | nation | business | technology | science | health | sports | entertainment", "technology"),
        ("max / page", "Up to 10 per request on the free plan", "10"),
        ("apikey", "Your key", "<GNEWS_API_KEY>"),
    ],
    sample="""{
  "totalArticles": 1523,
  "articles": [
    {"title": "EU begins enforcing AI Act rules for general-purpose models", "source": "Reuters", "publishedAt": "2026-10-11T14:05:00Z"}
  ]
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://gnews.io/api/v4/search", params={"q": '"artificial intelligence" AND regulation', "lang": "en",
                                                      "max": 5, "apikey": key("GNEWS_API_KEY")})
    show({"totalArticles": d["totalArticles"], "articles": [
        {"title": a["title"], "source": a["source"]["name"], "publishedAt": a["publishedAt"]} for a in d["articles"]]})


if __name__ == "__main__":
    main()
