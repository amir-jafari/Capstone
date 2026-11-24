META = dict(
    name="Lobsters (tech link aggregator)",
    domain="Social Media",
    docs="https://lobste.rs/about",
    auth="None",
    signup="No registration. Append .json to most pages. A small, invitation-only, moderated community - a useful "
           "contrast to Hacker News.",
    provides=["Hottest and newest stories with score, comment count, tags, submitter, timestamps",
              "Stories by tag, full comment threads (/s/{id}.json), user pages"],
    ideas="Comparing tech-community attention with Hacker News, tag-based topic trends, comment-tree analysis.",
    endpoint="GET https://lobste.rs/{hottest | newest | t/{tag} | s/{short_id}}.json",
    params=[
        ("hottest.json · newest.json", "Front page / newest stories (page/{n}.json for more)", "—"),
        ("t/{tag}.json", "Stories with a tag", "t/ai.json"),
        ("s/{short_id}.json", "One story with all comments", "—"),
        ("story fields", "title, url, score, comment_count, tags, created_at, submitter_user", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    hot = get("https://lobste.rs/hottest.json")
    show({"front_page_stories": len(hot), "tag_counts": Counter(t for s in hot for t in s["tags"]).most_common(6),
          "top": [{"title": s["title"], "score": s["score"], "comments": s["comment_count"], "tags": s["tags"]}
                  for s in hot[:4]]})


if __name__ == "__main__":
    main()
