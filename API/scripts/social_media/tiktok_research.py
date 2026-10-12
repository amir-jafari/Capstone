META = dict(
    name="TikTok Research API",
    domain="Social Media",
    docs="https://developers.tiktok.com/doc/research-api-get-started",
    auth="OAuth2 client credentials (approved researchers only)",
    env="TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET",
    signup="Requires an approved application: non-profit academic researchers (and some others) in eligible regions apply "
           "at https://developers.tiktok.com/products/research-api with a research proposal; approval takes weeks. Plan "
           "for this early in a capstone, or choose another platform.",
    provides=["Public videos by keyword, hashtag, region, date: description, views, likes, comments, shares, music, creator",
              "Comments on videos, public user info, followers/following, liked and pinned videos",
              "Date windows of up to 30 days per query; daily quota"],
    ideas="Short-video discourse on health or politics, hashtag diffusion, engagement modeling.",
    endpoint="POST https://open.tiktokapis.com/v2/research/video/query/?fields=...",
    params=[
        ("fields (query string)", "id, video_description, create_time, region_code, view_count, like_count, comment_count, share_count, hashtag_names", "id,video_description,view_count"),
        ("query (body)", "{and/or/not: [{operation: EQ | IN, field_name, field_values}]}", "{\"and\": [{\"operation\": \"IN\", \"field_name\": \"region_code\", \"field_values\": [\"US\"]}]}"),
        ("start_date / end_date (body)", "YYYYMMDD, at most 30 days apart", "20260901 / 20260930"),
        ("max_count / cursor / search_id (body)", "Up to 100 per page; continue with cursor and search_id", "100"),
        ("token", "POST https://open.tiktokapis.com/v2/oauth/token/ with client_key, client_secret, grant_type=client_credentials", "—"),
    ],
    sample="""[
  {"id": 7551234567890123456, "video_description": "3 data science projects that got me hired #datascience #python",
   "view_count": 182344, "like_count": 15210, "region_code": "US", "create_time": 1758912000}
]""",
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, key, show


def main():
    tok = requests.post("https://open.tiktokapis.com/v2/oauth/token/", timeout=30, data={
        "client_key": key("TIKTOK_CLIENT_KEY"), "client_secret": key("TIKTOK_CLIENT_SECRET"),
        "grant_type": "client_credentials"}).json()["access_token"]
    body = {"query": {"and": [{"operation": "IN", "field_name": "region_code", "field_values": ["US"]},
                              {"operation": "EQ", "field_name": "hashtag_name", "field_values": ["datascience"]}]},
            "start_date": "20260901", "end_date": "20260930", "max_count": 10}
    r = requests.post("https://open.tiktokapis.com/v2/research/video/query/", timeout=60, json=body,
                      params={"fields": "id,video_description,view_count,like_count,region_code,create_time"},
                      headers={"Authorization": f"Bearer {tok}", "User-Agent": USER_AGENT})
    r.raise_for_status()
    show(r.json()["data"]["videos"])


if __name__ == "__main__":
    main()
