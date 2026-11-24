META = dict(
    name="Wikimedia REST API (Pageviews)",
    domain="News & Media",
    docs="https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/",
    auth="None (descriptive User-Agent required)",
    signup="No registration. Keep under ~100 req/s and send a User-Agent with contact info.",
    provides=["Daily/monthly pageviews per article, per project, per country",
              "Most-viewed articles per day", "Article summaries, revisions, edit counts"],
    ideas="Public attention as a proxy signal: predicting elections, disease outbreaks, stock volatility.",
    endpoint="GET https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}",
    params=[
        ("project", "Wiki project", "en.wikipedia"),
        ("access", "all-access | desktop | mobile-app | mobile-web", "all-access"),
        ("agent", "all-agents | user | spider | automated", "user"),
        ("article", "Article title with underscores (URL-encoded)", "Artificial_intelligence"),
        ("granularity", "daily | monthly", "daily"),
        ("start / end", "YYYYMMDD (or YYYYMMDDHH)", "20260901 / 20260907"),
        ("top/{project}/{access}/{year}/{month}/{day}", "Sibling endpoint: 1,000 most-viewed articles that day", "top/en.wikipedia/all-access/2026/09/01"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           "en.wikipedia/all-access/user/Artificial_intelligence/daily/20260901/20260907")
    show([{"date": i["timestamp"][:8], "views": i["views"]} for i in get(url)["items"]], max_items=7)


if __name__ == "__main__":
    main()
