META = dict(
    name="Spotify Web API",
    domain="Culture, Books & Entertainment",
    docs="https://developer.spotify.com/documentation/web-api",
    auth="OAuth2 client credentials (client ID + secret)",
    env="SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET",
    signup="Create an app at https://developer.spotify.com/dashboard to get a client ID and secret. Apps start in "
           "development mode with limits. Since Nov 2024 new apps can no longer call audio-features, audio-analysis or "
           "recommendations - plan around search, artists, albums, tracks and playlists.",
    provides=["Search artists, albums, tracks, playlists, shows, episodes, audiobooks",
              "Artist popularity (0-100), followers, genres; album tracks and release dates; track popularity",
              "Public playlists and podcast (show) metadata"],
    ideas="Popularity vs. release timing, genre landscapes, podcast catalog analysis (audio-features no longer available).",
    endpoint="GET https://api.spotify.com/v1/{resource}  (token: POST https://accounts.spotify.com/api/token)",
    params=[
        ("search?q=&type=", "type: artist, album, track, playlist, show, episode (comma list); field filters year:, genre:, tag:new", "q=genre:jazz year:2025&type=album"),
        ("artists/{id} · artists/{id}/albums · artists/{id}/top-tracks?market=", "Artist details, discography, top tracks", "market=US"),
        ("albums/{id}/tracks · tracks/{id}", "Album tracks; track details incl. popularity", "—"),
        ("limit / offset / market", "Max 50 per page; country market", "10 / 0 / US"),
        ("grant_type=client_credentials", "Token request (Basic auth with client ID:secret); token lasts 1 hour", "—"),
    ],
    sample="""{
  "query": "genre:jazz year:2025",
  "albums": [
    {"name": "Example Jazz Album", "artist": "Example Trio", "release_date": "2025-03-14", "total_tracks": 9}
  ]
}""",
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    tok = requests.post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"},
                        auth=(key("SPOTIFY_CLIENT_ID"), key("SPOTIFY_CLIENT_SECRET")), timeout=30).json()["access_token"]
    d = get("https://api.spotify.com/v1/search", headers={"Authorization": f"Bearer {tok}"},
            params={"q": "genre:jazz year:2025", "type": "album", "limit": 5, "market": "US"})["albums"]["items"]
    show({"query": "genre:jazz year:2025", "albums": [
        {"name": a["name"], "artist": a["artists"][0]["name"], "release_date": a["release_date"],
         "total_tracks": a["total_tracks"]} for a in d]})


if __name__ == "__main__":
    main()
