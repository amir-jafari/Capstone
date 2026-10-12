META = dict(
    name="Reddit Data API",
    domain="Social Media",
    docs="https://www.reddit.com/dev/api/",
    auth="OAuth2 (client id + secret, 'script' app)",
    env="REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET",
    signup="1) Log in, go to https://www.reddit.com/prefs/apps -> 'create another app' -> type **script**, "
           "redirect uri `http://localhost:8080`. 2) The string under the app name is the client id; "
           "'secret' is the secret. 3) Since 2024 you must also accept the Data API terms "
           "(https://support.reddithelp.com/hc/en-us/articles/14945211791892). Free tier: 100 QPM per client. "
           "Unauthenticated `.json` URLs now return 403 for most scripts.",
    provides=["Posts per subreddit (hot/new/top), title, selftext, score, upvote ratio, num_comments",
              "Full comment trees", "Subreddit metadata, user public history", "Search across Reddit"],
    ideas="Opinion mining, community detection, mental-health discourse, meme-stock sentiment (r/wallstreetbets).",
    sample="""[
  {
    "title": "What's one data-science skill that paid off the most in your career?",
    "score": 812,
    "num_comments": 241,
    "upvote_ratio": 0.97,
    "created_utc": 1760000000.0
  },
  {
    "title": "Weekly Entering & Transitioning Thread",
    "score": 15,
    "num_comments": 98,
    "upvote_ratio": 0.9,
    "created_utc": 1759800000.0
  }
]""",
    endpoint="GET https://oauth.reddit.com/r/{subreddit}/{sort}  (token: POST https://www.reddit.com/api/v1/access_token)",
    params=[
        ("subreddit", "Subreddit name (path)", "datascience"),
        ("sort", "hot | new | top | rising | controversial (path)", "top"),
        ("t", "Time window for top/controversial: hour | day | week | month | year | all", "week"),
        ("limit", "Items per request (max 100)", "5"),
        ("after / before", "Fullname cursor for paging (from the previous response)", "t3_1abc23"),
        ("grant_type", "Token request body; client_credentials = read-only app access", "client_credentials"),
        ("/r/{sub}/comments/{id}", "Sibling endpoint: a post with its full comment tree", "/r/datascience/comments/1abc23"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import requests

from _common import USER_AGENT, get, key, show


def main():
    # Application-only OAuth (client_credentials) is enough for reading public data.
    tok = requests.post("https://www.reddit.com/api/v1/access_token",
                        auth=(key("REDDIT_CLIENT_ID"), key("REDDIT_CLIENT_SECRET")),
                        data={"grant_type": "client_credentials"},
                        headers={"User-Agent": USER_AGENT}, timeout=30).json()["access_token"]
    posts = get("https://oauth.reddit.com/r/datascience/top", params={"t": "week", "limit": 5},
                headers={"Authorization": f"bearer {tok}"})["data"]["children"]
    show([{k: p["data"][k] for k in ("title", "score", "num_comments", "upvote_ratio", "created_utc")}
          for p in posts])


if __name__ == "__main__":
    main()
