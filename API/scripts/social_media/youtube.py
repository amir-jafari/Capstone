META = dict(
    name="YouTube Data API v3",
    domain="Social Media",
    docs="https://developers.google.com/youtube/v3/docs",
    auth="API key (Google Cloud)",
    env="YOUTUBE_API_KEY",
    signup="1) https://console.cloud.google.com -> create project. 2) APIs & Services -> Library -> enable "
           "'YouTube Data API v3'. 3) Credentials -> Create credentials -> API key. "
           "Free quota: 10,000 units/day (a search costs 100, a videos.list costs 1).",
    provides=["Video search; title, description, channel, publish date", "Statistics: views, likes, comment counts",
              "Comment threads (text, likes, replies)", "Channel subscribers & uploads"],
    ideas="Comment sentiment, misinformation spread, creator-economy analytics.",
    sample="""[
  {
    "title": "But what is a neural network? | Deep learning chapter 1",
    "channel": "3Blue1Brown",
    "views": "21534876",
    "likes": "498211",
    "comments": "14232"
  }
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    k = key("YOUTUBE_API_KEY")
    ids = [i["id"]["videoId"] for i in get("https://www.googleapis.com/youtube/v3/search", params={
        "part": "snippet", "q": "neural network", "type": "video", "maxResults": 3, "key": k})["items"]]
    vids = get("https://www.googleapis.com/youtube/v3/videos",
               params={"part": "snippet,statistics", "id": ",".join(ids), "key": k})["items"]
    show([{"title": v["snippet"]["title"], "channel": v["snippet"]["channelTitle"],
           "views": v["statistics"].get("viewCount"), "likes": v["statistics"].get("likeCount"),
           "comments": v["statistics"].get("commentCount")} for v in vids])


if __name__ == "__main__":
    main()
