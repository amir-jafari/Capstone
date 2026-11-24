META = dict(
    name="GDELT DOC 2.0",
    domain="News & Media",
    docs="https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
    auth="None",
    signup="No registration. Be polite: ~1 request every 5 seconds.",
    provides=[
        "Global news articles (65+ languages) from the last ~3 months matching a query",
        "Article URL, title, domain, source country, language, timestamp",
        "Timeline modes: volume / tone (sentiment) of coverage over time",
    ],
    ideas="Media framing & sentiment around elections, conflicts, policy events; event detection.",
    endpoint="GET https://api.gdeltproject.org/api/v2/doc/doc",
    params=[
        ("query", "Search terms plus operators: sourcecountry:, sourcelang:, domain:, theme:, tone<, \"exact phrase\"", "\"artificial intelligence\" sourcecountry:US"),
        ("mode", "artlist | timelinevol | timelinetone | timelinesourcecountry | tonechart", "artlist"),
        ("format", "json | csv | html | rss", "json"),
        ("maxrecords", "Articles returned in artlist mode (max 250)", "5"),
        ("timespan", "Relative window ending now (min, h, d, w, m)", "1w"),
        ("startdatetime / enddatetime", "Absolute window, YYYYMMDDHHMMSS (last ~3 months)", "20260901000000"),
        ("sort", "datedesc | dateasc | tonedesc | toneasc | hybridrel", "datedesc"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    data = get("https://api.gdeltproject.org/api/v2/doc/doc", params={
        "query": '"artificial intelligence" sourcecountry:US',
        "mode": "artlist", "maxrecords": 5, "format": "json", "sort": "datedesc",
    })
    show([{k: a[k] for k in ("title", "domain", "seendate", "language", "sourcecountry")}
          for a in data.get("articles", [])])


if __name__ == "__main__":
    main()
