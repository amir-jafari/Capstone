META = dict(
    name="X (Twitter) API v2",
    domain="Social Media",
    docs="https://docs.x.com/x-api/introduction",
    auth="Bearer token (paid access for reading)",
    env="X_BEARER_TOKEN",
    signup="Not free for research: the free tier is essentially write-only, and searching or reading posts needs a "
           "paid plan - check https://developer.x.com for current pricing before planning a project around it. The "
           "academic research track was discontinued in 2023. Consider Bluesky or Mastodon (free) instead.",
    provides=["Recent search (last 7 days) and full-archive search (higher tiers) with the X query language",
              "Post text, author, created_at, public metrics (likes, reposts, replies, quotes, views), language, entities",
              "Users, followers/following, lists, spaces"],
    ideas="Only if you have access: discourse analysis, information diffusion, public-metrics modeling.",
    endpoint="GET https://api.x.com/2/{tweets/search/recent | tweets/search/all | users/by/username/{u}}",
    params=[
        ("query", "Search operators: keywords, from:, lang:, -is:retweet, has:links ...", "\"data science\" lang:en -is:retweet"),
        ("max_results", "10-100 per page", "10"),
        ("tweet.fields", "created_at, public_metrics, lang, author_id, entities ...", "created_at,public_metrics"),
        ("start_time / end_time / next_token", "Time window and paging", "—"),
        ("Authorization (header)", "Bearer <token>", "Bearer <X_BEARER_TOKEN>"),
    ],
    sample="""[
  {"created_at": "2026-10-11T18:22:05Z", "text": "New blog post: what I learned from a year of data science interviews...",
   "likes": 214, "reposts": 37, "replies": 12}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.x.com/2/tweets/search/recent", headers={"Authorization": f"Bearer {key('X_BEARER_TOKEN')}"},
            params={"query": '"data science" lang:en -is:retweet', "max_results": 10,
                    "tweet.fields": "created_at,public_metrics"})
    show([{"created_at": t["created_at"], "text": t["text"], "likes": t["public_metrics"]["like_count"],
           "reposts": t["public_metrics"]["retweet_count"], "replies": t["public_metrics"]["reply_count"]}
          for t in d.get("data", [])])


if __name__ == "__main__":
    main()
