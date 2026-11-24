META = dict(
    name="The Guardian Open Platform",
    domain="News & Media",
    docs="https://open-platform.theguardian.com/documentation/",
    auth="API key (query `api-key`)",
    env="GUARDIAN_API_KEY",
    signup="Register for a free 'Developer key' at https://open-platform.theguardian.com/access/ "
           "(emailed instantly). 1 call/sec, 500 calls/day, non-commercial.",
    provides=["Search 2M+ articles back to 1999 with full body text (`show-fields=bodyText`)",
              "Section, tags (keywords, contributors), publication date",
              "Great for longitudinal corpora — the archive is free and complete"],
    ideas="Long-run framing of immigration/climate in the press, NLP corpora with full text.",
    sample="""{
  "total": 18432,
  "results": [
    {
      "webTitle": "What the polls say three weeks before the midterms",
      "sectionName": "US news",
      "webPublicationDate": "2026-10-10T11:00:12Z",
      "wordcount": "1243",
      "webUrl": "https://www.theguardian.com/us-news/2026/oct/10/..."
    }
  ]
}""",
    endpoint="GET https://content.guardianapis.com/search",
    params=[
        ("q", "Query with AND / OR / NOT and \"phrases\"", "election"),
        ("section", "Section ID (list via /sections)", "politics"),
        ("tag", "Tag ID(s), comma = AND, | = OR (list via /tags)", "politics/politics"),
        ("from-date / to-date", "Publication date range, YYYY-MM-DD", "2020-01-01"),
        ("order-by", "newest | oldest | relevance", "newest"),
        ("show-fields", "Extra fields: bodyText, body, headline, byline, wordcount, thumbnail", "wordcount,bodyText"),
        ("page-size / page", "Results per page (max 200) and page number", "5"),
        ("api-key", "Your key (required)", "<GUARDIAN_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    data = get("https://content.guardianapis.com/search", params={
        "q": "election", "page-size": 5, "order-by": "newest",
        "show-fields": "wordcount", "api-key": key("GUARDIAN_API_KEY"),
    })["response"]
    show({"total": data["total"], "results": [
        {"webTitle": r["webTitle"], "sectionName": r["sectionName"],
         "webPublicationDate": r["webPublicationDate"], "wordcount": r["fields"]["wordcount"], "webUrl": r["webUrl"]}
        for r in data["results"]]})


if __name__ == "__main__":
    main()
