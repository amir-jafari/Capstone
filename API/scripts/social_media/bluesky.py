META = dict(
    name="Bluesky (AT Protocol) Public AppView",
    domain="Social Media",
    docs="https://docs.bsky.app/docs/category/http-reference",
    auth="None for public read endpoints (search needs an app password)",
    signup="Reading public profiles/feeds needs nothing. For search, create an account and an app password at "
           "Settings -> Privacy and security -> App passwords, then call `com.atproto.server.createSession`. "
           "The full firehose is open too (Jetstream: wss://jetstream2.us-east.bsky.network).",
    provides=["Profiles: followers, following, post counts, bio", "Author feeds: post text, likes, reposts, replies",
              "Follower graphs", "Real-time firehose of every public post"],
    ideas="The most open large social network for research now that X/Twitter API is paid; network & discourse analysis.",
    endpoint="GET https://public.api.bsky.app/xrpc/{method}",
    params=[
        ("app.bsky.actor.getProfile?actor=", "Handle or DID", "bsky.app"),
        ("app.bsky.feed.getAuthorFeed?actor=&limit=&cursor=", "A user's posts; limit 1-100, cursor for paging", "limit=3"),
        ("filter (getAuthorFeed)", "posts_with_replies | posts_no_replies | posts_with_media | posts_and_author_threads", "posts_no_replies"),
        ("app.bsky.graph.getFollowers?actor=&limit=&cursor=", "Follower list (also getFollows)", "actor=bsky.app"),
        ("app.bsky.feed.getPostThread?uri=", "A post and its reply tree (at:// URI)", "at://did:plc:…/app.bsky.feed.post/…"),
        ("app.bsky.feed.searchPosts?q=&sort=", "Post search, sort top | latest (requires an authenticated session)", "q=climate"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://public.api.bsky.app/xrpc"


def main():
    prof = get(f"{BASE}/app.bsky.actor.getProfile", params={"actor": "bsky.app"})
    feed = get(f"{BASE}/app.bsky.feed.getAuthorFeed", params={"actor": prof["did"], "limit": 3})["feed"]
    show({
        "profile": {k: prof.get(k) for k in ("handle", "displayName", "followersCount", "postsCount")},
        "recent_posts": [{"text": f["post"]["record"].get("text"), "likes": f["post"].get("likeCount"),
                          "reposts": f["post"].get("repostCount"), "createdAt": f["post"]["record"].get("createdAt")}
                         for f in feed],
    })


if __name__ == "__main__":
    main()
