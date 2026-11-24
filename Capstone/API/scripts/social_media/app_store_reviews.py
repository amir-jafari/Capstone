META = dict(
    name="Apple App Store customer reviews (RSS JSON)",
    domain="Social Media",
    docs="https://rss.marketingtools.apple.com/",
    auth="None",
    signup="No registration. Legacy RSS feed that still returns the 500 most recent reviews per app and country "
           "(10 pages x 50). Find the app ID in its App Store URL (.../id6448311069).",
    provides=["Review title, text, star rating (1-5), app version, author, date, vote counts",
              "Per-country storefronts (us, gb, de, in, ...)", "Most-recent or most-helpful ordering"],
    ideas="App sentiment over versions (did an update anger users?), feature-request mining, rating prediction from text.",
    endpoint="GET https://itunes.apple.com/{country}/rss/customerreviews/page={n}/id={appId}/sortby={order}/json",
    params=[
        ("country", "Storefront code", "us"),
        ("page", "1-10 (50 reviews each)", "1"),
        ("appId", "Numeric app ID", "6448311069 (ChatGPT)"),
        ("sortby", "mostrecent | mosthelpful", "mostrecent"),
        ("entry fields", "im:rating, title, content, im:version, author.name, updated", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    entries = []
    for page in (1, 2):
        d = get(f"https://itunes.apple.com/us/rss/customerreviews/page={page}/id=6448311069/sortby=mostrecent/json")
        entries += [e for e in d["feed"].get("entry", []) if "im:rating" in e]
    stars = Counter(int(e["im:rating"]["label"]) for e in entries)
    show({"app": "ChatGPT (US App Store)", "reviews_read": len(entries), "star_distribution": dict(sorted(stars.items())),
          "average_stars": round(sum(k * v for k, v in stars.items()) / len(entries), 2),
          "latest": [{"stars": int(e["im:rating"]["label"]), "version": e["im:version"]["label"], "title": e["title"]["label"],
                      "date": e["updated"]["label"][:10]} for e in entries[:3]]})


if __name__ == "__main__":
    main()
