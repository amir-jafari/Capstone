META = dict(
    name="Last.fm API (listening data & tags)",
    domain="Culture, Books & Entertainment",
    docs="https://www.last.fm/api",
    auth="API key (query `api_key`)",
    env="LASTFM_API_KEY",
    signup="Create an API account at https://www.last.fm/api/account/create (instant). Non-commercial use; keep to "
           "about 5 requests/second.",
    provides=["Global and country charts (top artists/tracks) with play and listener counts",
              "Crowd-sourced tags (genres, moods), similar artists/tracks, artist bios",
              "Users' public listening histories (scrobbles)"],
    ideas="Music recommendation (similarity graphs), mood/genre tagging, listening-behavior analysis.",
    endpoint="GET https://ws.audioscrobbler.com/2.0/?method={method}",
    params=[
        ("method", "chart.getTopArtists | geo.getTopTracks | artist.getSimilar | track.getTopTags | tag.getTopTracks | user.getRecentTracks", "geo.getTopTracks"),
        ("artist / track / tag / country / user", "Method-specific arguments", "country=united states"),
        ("limit / page", "Paging", "5"),
        ("format", "json | xml", "json"),
        ("api_key", "Your key", "<LASTFM_API_KEY>"),
    ],
    sample="""{
  "top_artists": [
    {"name": "Taylor Swift", "listeners": "6418811", "playcount": "1823456712"},
    {"name": "The Weeknd", "listeners": "7012345", "playcount": "1213456789"}
  ]
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://ws.audioscrobbler.com/2.0/", params={
        "method": "chart.gettopartists", "api_key": key("LASTFM_API_KEY"), "format": "json", "limit": 5})
    show({"top_artists": [{k: a[k] for k in ("name", "listeners", "playcount")} for a in d["artists"]["artist"]]})


if __name__ == "__main__":
    main()
