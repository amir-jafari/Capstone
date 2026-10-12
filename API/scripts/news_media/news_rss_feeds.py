META = dict(
    name="Publisher RSS feeds (BBC, NPR, and others)",
    domain="News & Media",
    docs="https://www.bbc.co.uk/news/10628494 (BBC feed list); NPR feeds follow feeds.npr.org/{id}/rss.xml",
    auth="None",
    signup="No registration. Most outlets publish RSS/Atom feeds by section. Terms vary by publisher (often personal, "
           "non-commercial use with attribution) - check each one before redistributing. Google News RSS is excluded "
           "here because its feed terms limit use to personal feed readers.",
    provides=["Latest headlines with links, short summaries, publication times, sometimes images and categories",
              "Section-level feeds (world, politics, business, technology, health...)", "Easy to poll on a schedule to build your own news archive"],
    ideas="Building a dated headline corpus for framing/sentiment studies, comparing story selection across outlets.",
    endpoint="GET {feed URL}  (RSS 2.0 XML)",
    params=[
        ("BBC", "feeds.bbci.co.uk/news/{section}/rss.xml: world, politics, business, technology, health, science_and_environment", "world"),
        ("NPR", "feeds.npr.org/{id}/rss.xml: 1001 news, 1014 politics, 1006 business, 1019 technology, 1128 health", "1001"),
        ("item fields", "title, link, description, pubDate (plus category / media tags on some feeds)", "—"),
        ("polling", "Feeds hold only recent items - poll regularly and store results to build history", "every 30-60 min"),
    ],
)
import email.utils
import xml.etree.ElementTree as ET

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

FEEDS = {"BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml", "NPR News": "https://feeds.npr.org/1001/rss.xml"}


def main():
    out = {}
    for name, url in FEEDS.items():
        items = ET.fromstring(get(url, as_json=False).encode()).findall("./channel/item")
        out[name] = {"items": len(items), "latest": [
            {"title": (i.findtext("title") or "").strip(),
             "published": email.utils.parsedate_to_datetime(i.findtext("pubDate")).isoformat() if i.findtext("pubDate") else None}
            for i in items[:3]]}
    show(out)


if __name__ == "__main__":
    main()
