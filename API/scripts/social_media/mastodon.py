META = dict(
    name="Mastodon",
    domain="Social Media",
    docs="https://docs.joinmastodon.org/methods/",
    auth="None for public timelines/trends (per-instance)",
    signup="No key for public endpoints. For more, register an app on any instance (Preferences -> Development).",
    provides=["Trending hashtags with daily usage history", "Public / local timelines, hashtag timelines",
              "Account info, statuses (HTML content, boosts, favourites)"],
    ideas="Decentralized-social-network studies, hashtag diffusion across instances.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    tags = get("https://mastodon.social/api/v1/trends/tags", params={"limit": 5})
    show([{"tag": t["name"], "uses_today": t["history"][0]["uses"], "accounts_today": t["history"][0]["accounts"]}
          for t in tags])


if __name__ == "__main__":
    main()
