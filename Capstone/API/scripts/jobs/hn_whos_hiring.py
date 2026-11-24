META = dict(
    name="Hacker News 'Who is hiring?' (via Algolia HN Search)",
    domain="Jobs & Labor",
    docs="https://hn.algolia.com/api",
    auth="None",
    signup="No registration. A monthly thread since 2011 where tech companies post jobs as comments - a long-running, "
           "free dataset of tech hiring.",
    provides=["Every 'Ask HN: Who is hiring?' thread (posted by user whoishiring) with its date",
              "All comments (job posts) per thread: company, role, location, REMOTE/ONSITE, salary text, tech stack",
              "Full-text search across HN stories and comments with date filters"],
    ideas="Tech-hiring cycle index (posts per month), remote share over time, programming-language demand trends.",
    endpoint="GET https://hn.algolia.com/api/v1/{search | search_by_date | items/{id}}",
    params=[
        ("query", "Search text", "\"Who is hiring\""),
        ("tags", "story | comment | author_{user} | story_{id} (comma = AND)", "story,author_whoishiring"),
        ("numericFilters", "created_at_i>..., points>...", "created_at_i>1735689600"),
        ("hitsPerPage / page", "Max 1,000 per page", "12"),
        ("tags=comment,story_{id}", "All top-level job posts in one thread", "comment,story_45000000"),
        ("/items/{id}", "A thread with its full comment tree", "—"),
    ],
)
import datetime as dt

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://hn.algolia.com/api/v1"


def main():
    threads = get(f"{BASE}/search_by_date", params={"query": '"Who is hiring"', "tags": "story,author_whoishiring",
                                                     "hitsPerPage": 12})["hits"]
    threads = [t for t in threads if t["title"].startswith("Ask HN: Who is hiring?")][:6]
    latest = threads[0]
    posts = get(f"{BASE}/search", params={"tags": f"comment,story_{latest['objectID']}", "hitsPerPage": 1000})["hits"]
    top = [p for p in posts if p.get("parent_id") == int(latest["objectID"])]  # top-level comments = job posts
    show({"recent_threads": [{"title": t["title"], "comments": t["num_comments"]} for t in threads],
          "latest_thread_job_posts": len(top),
          "share_mentioning_REMOTE": round(sum("remote" in (p.get("comment_text") or "").lower() for p in top) / max(len(top), 1), 2),
          "posted": dt.datetime.fromtimestamp(latest["created_at_i"]).date().isoformat()})


if __name__ == "__main__":
    main()
