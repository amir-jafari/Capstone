META = dict(
    name="New York Times APIs",
    domain="News & Media",
    docs="https://developer.nytimes.com/apis",
    auth="API key (query `api-key`)",
    env="NYT_API_KEY",
    signup="Create an account at https://developer.nytimes.com -> Apps -> New App -> enable "
           "'Article Search API' (and Archive/Top Stories as needed). 10 req/min, 4,000 req/day.",
    provides=["Article Search: headline, abstract, keywords, section, byline, pub_date (back to 1851)",
              "Archive API: every article metadata for a given month",
              "Top Stories, Most Popular, Books best-seller lists"],
    ideas="150+ years of headline data for historical discourse analysis; best-seller trends.",
    sample="""[
  {
    "headline": "Supreme Court Opens New Term With Major Cases on Agency Power",
    "pub_date": "2026-10-06T09:00:05+0000",
    "section_name": "U.S.",
    "keywords": ["Supreme Court (US)", "Law and Legislation", "Regulation and Deregulation of Industry"]
  }
]""",
    endpoint="GET https://api.nytimes.com/svc/search/v2/articlesearch.json",
    params=[
        ("q", "Full-text query", "supreme court"),
        ("fq", "Filter query on fields such as section, type of material, byline (field names per the docs)", "—"),
        ("begin_date / end_date", "Date range, YYYYMMDD", "20260101"),
        ("sort", "newest | oldest | relevance", "newest"),
        ("page", "Page number, 10 results each (max page 100)", "0"),
        ("api-key", "Your key (required)", "<NYT_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    docs = get("https://api.nytimes.com/svc/search/v2/articlesearch.json",
               params={"q": "supreme court", "sort": "newest", "api-key": key("NYT_API_KEY")})["response"]["docs"]
    show([{"headline": d["headline"]["main"], "pub_date": d["pub_date"], "section_name": d.get("section_name"),
           "keywords": [k["value"] for k in d.get("keywords", [])][:3]} for d in docs])


if __name__ == "__main__":
    main()
