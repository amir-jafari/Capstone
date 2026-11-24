META = dict(
    name="Hacker News (Firebase API)",
    domain="News & Media",
    docs="https://github.com/HackerNews/API",
    auth="None",
    signup="No registration, no rate limit published.",
    provides=["Top / new / best / Ask / Show story IDs", "Story title, URL, score, author, comment count, time",
              "Full comment trees (item `kids`)", "User profiles (karma, submissions)"],
    ideas="Tech-trend detection, comment toxicity / sentiment, predicting story popularity.",
    endpoint="GET https://hacker-news.firebaseio.com/v0/{resource}.json",
    params=[
        ("topstories / newstories / beststories", "Up to 500 current story IDs", "topstories"),
        ("askstories / showstories / jobstories", "Up to 200 IDs of each story type", "askstories"),
        ("item/{id}", "A story, comment, poll or job (`kids` = child comment IDs)", "item/8863"),
        ("user/{id}", "User profile: karma, created, submitted item IDs", "user/pg"),
        ("maxitem", "Largest item ID so far (walk backwards to crawl everything)", "maxitem"),
        ("updates", "Recently changed items and profiles", "updates"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://hacker-news.firebaseio.com/v0"


def main():
    ids = get(f"{BASE}/topstories.json")[:5]
    stories = [get(f"{BASE}/item/{i}.json") for i in ids]
    show([{k: s.get(k) for k in ("title", "url", "score", "by", "descendants")} for s in stories])


if __name__ == "__main__":
    main()
