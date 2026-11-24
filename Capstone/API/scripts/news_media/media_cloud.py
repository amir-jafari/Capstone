META = dict(
    name="Media Cloud (news research platform)",
    domain="News & Media",
    docs="https://github.com/mediacloud/api-tutorial-notebooks",
    auth="API token",
    env="MEDIACLOUD_API_KEY",
    signup="Create a free account at https://search.mediacloud.org and copy the API key from your profile. Python "
           "client: `pip install mediacloud`. Built for academic research on online news (MIT/UMass/Northeastern).",
    provides=["Search millions of online news stories per day from curated source collections (US national, state/local, global by country)",
              "Story counts over time (attention), story lists with URLs, titles, publish dates, media source",
              "Source and collection metadata"],
    ideas="Attention to issues across national vs. local news, framing studies, media ecosystem mapping.",
    endpoint="Python: mediacloud.api.SearchApi(token).{story_count | story_count_over_time | story_list}",
    params=[
        ("query", "Boolean query (Lucene syntax)", "\"climate change\" AND policy"),
        ("start_date / end_date", "datetime.date objects", "2026-09-01 / 2026-09-30"),
        ("collection_ids", "Source collections, e.g. 34412234 US national news", "[34412234]"),
        ("source_ids", "Specific outlets", "—"),
        ("story_list pagination_token", "Page through stories", "—"),
    ],
    sample="""{
  "query": "\\"climate change\\"",
  "collection": "US national news",
  "stories_matching": 4128,
  "share_of_all_stories_pct": 0.92,
  "daily_counts_first_3_days": [{"date": "2026-09-01", "count": 118}, {"date": "2026-09-02", "count": 149}, {"date": "2026-09-03", "count": 152}]
}""",
)
import datetime as dt

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import key, show


def main():
    token = key("MEDIACLOUD_API_KEY")
    import mediacloud.api  # pip install mediacloud (only needed for this example)
    api = mediacloud.api.SearchApi(token)
    start, end, us_national = dt.date(2026, 9, 1), dt.date(2026, 9, 30), [34412234]
    counts = api.story_count('"climate change"', start, end, collection_ids=us_national)
    daily = api.story_count_over_time('"climate change"', start, end, collection_ids=us_national)
    show({"query": '"climate change"', "collection": "US national news", "stories_matching": counts["relevant"],
          "share_of_all_stories_pct": round(100 * counts["relevant"] / counts["total"], 2),
          "daily_counts_first_3_days": [{"date": str(r["date"]), "count": r["count"]} for r in daily[:3]]})


if __name__ == "__main__":
    main()
