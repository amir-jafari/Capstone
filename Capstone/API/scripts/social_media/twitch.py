META = dict(
    name="Twitch Helix API",
    domain="Social Media",
    docs="https://dev.twitch.tv/docs/api/reference",
    auth="OAuth2 app access token (client ID + secret)",
    env="TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET",
    signup="Register an application at https://dev.twitch.tv/console/apps (requires 2FA on your Twitch account). Get an "
           "app token from https://id.twitch.tv/oauth2/token with grant_type=client_credentials; send both Client-Id and "
           "Authorization: Bearer headers. 800 points/minute.",
    provides=["Live streams with viewer counts, game/category, language, start time, tags",
              "Top games/categories by viewers, channel info, clips, videos (VODs), schedules",
              "Chat is separate (IRC/EventSub)"],
    ideas="Live-streaming attention economy, game popularity vs. sales (SteamSpy), streamer growth dynamics.",
    endpoint="GET https://api.twitch.tv/helix/{streams | games/top | games | clips | videos | channels}",
    params=[
        ("streams?first=&game_id=&language=", "Live streams sorted by viewers (first max 100)", "first=20&language=en"),
        ("games/top?first=", "Top categories by current viewers", "first=10"),
        ("games?name=", "Look up a game/category ID", "name=Chess"),
        ("clips?game_id=&started_at=", "Top clips", "—"),
        ("after", "Pagination cursor", "—"),
        ("headers", "Client-Id: <id>, Authorization: Bearer <app token>", "—"),
    ],
    sample="""{
  "top_categories": ["Just Chatting", "Grand Theft Auto V", "League of Legends", "Counter-Strike", "Valorant"],
  "top_live_streams": [
    {"user": "example_streamer", "game": "Just Chatting", "viewers": 84211, "language": "en"}
  ]
}""",
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    cid, secret = key("TWITCH_CLIENT_ID"), key("TWITCH_CLIENT_SECRET")
    tok = requests.post("https://id.twitch.tv/oauth2/token", timeout=30, params={
        "client_id": cid, "client_secret": secret, "grant_type": "client_credentials"}).json()["access_token"]
    hdr = {"Client-Id": cid, "Authorization": f"Bearer {tok}"}
    games = get("https://api.twitch.tv/helix/games/top", headers=hdr, params={"first": 5})["data"]
    streams = get("https://api.twitch.tv/helix/streams", headers=hdr, params={"first": 5})["data"]
    show({"top_categories": [g["name"] for g in games], "top_live_streams": [
        {"user": s["user_name"], "game": s["game_name"], "viewers": s["viewer_count"], "language": s["language"]} for s in streams]})


if __name__ == "__main__":
    main()
