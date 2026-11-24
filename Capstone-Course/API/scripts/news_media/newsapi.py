META = dict(
    name="NewsAPI.org",
    domain="News & Media",
    docs="https://newsapi.org/docs",
    auth="API key (header `X-Api-Key`)",
    env="NEWSAPI_KEY",
    signup="Register at https://newsapi.org/register -> key shown on dashboard. Free 'Developer' plan: "
           "100 requests/day, articles up to 1 month old, development use only.",
    provides=["Headlines by country / category / source (~150k sources)",
              "Full-text search across articles (`/v2/everything`)",
              "Title, description, source, author, URL, image, publishedAt, content snippet"],
    ideas="News topic modeling, headline sentiment vs. stock moves, source bias comparison.",
    sample="""[
  {
    "source": "Reuters",
    "title": "Fed holds rates steady, signals patience on cuts",
    "publishedAt": "2026-10-10T18:42:00Z",
    "url": "https://www.reuters.com/markets/..."
  },
  {
    "source": "The Verge",
    "title": "...",
    "publishedAt": "2026-10-10T17:05:11Z",
    "url": "https://www.theverge.com/..."
  }
]""",
    endpoint="GET https://newsapi.org/v2/top-headlines  ·  GET https://newsapi.org/v2/everything",
    params=[
        ("country", "2-letter country code (top-headlines only)", "us"),
        ("category", "business | entertainment | general | health | science | sports | technology (top-headlines)", "business"),
        ("sources", "Comma-separated source IDs; can't be combined with country/category", "bbc-news,reuters"),
        ("q", "Keywords or phrase; /everything supports AND / OR / NOT", "\"climate change\" AND policy"),
        ("from / to", "ISO-8601 date range (/everything only)", "2026-10-01"),
        ("language", "2-letter language code (/everything)", "en"),
        ("sortBy", "relevancy | popularity | publishedAt (/everything)", "publishedAt"),
        ("pageSize / page", "Results per page (max 100) and page number", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    data = get("https://newsapi.org/v2/top-headlines",
               params={"country": "us", "category": "business", "pageSize": 5},
               headers={"X-Api-Key": key("NEWSAPI_KEY")})
    show([{"source": a["source"]["name"], "title": a["title"], "publishedAt": a["publishedAt"], "url": a["url"]}
          for a in data["articles"]])


if __name__ == "__main__":
    main()
