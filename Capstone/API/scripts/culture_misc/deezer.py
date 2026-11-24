META = dict(
    name="Deezer API (music charts & catalog)",
    domain="Culture, Books & Entertainment",
    docs="https://developers.deezer.com/api",
    auth="None for public data",
    signup="No registration for catalog and chart endpoints (50 requests / 5 seconds). OAuth only for user data. "
           "The docs page now asks you to log in with a (free) Deezer account.",
    provides=["Global and per-genre charts: top tracks, albums, artists, playlists, podcasts",
              "Track details: duration, BPM, rank (popularity), explicit flag, 30-second preview MP3",
              "Artists (fan counts, top tracks, related artists), albums, search"],
    ideas="Chart dynamics, audio analysis on preview clips, related-artist networks, explicit-content trends.",
    endpoint="GET https://api.deezer.com/{resource}",
    params=[
        ("chart/{genre_id}/{tracks | albums | artists | playlists}", "Charts; genre 0 = all (list via /genre)", "chart/0/tracks"),
        ("search?q=", "Search with advanced syntax: artist:\"\" track:\"\" album:\"\" bpm_min: dur_max:", "q=artist:\"daft punk\""),
        ("track/{id} · album/{id} · artist/{id}", "Details (track includes bpm, rank)", "track/3135556"),
        ("artist/{id}/top · artist/{id}/related", "An artist's top tracks / similar artists", "—"),
        ("limit / index", "Paging", "10 / 0"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    tracks = get("https://api.deezer.com/chart/0/tracks", params={"limit": 5})["data"]
    artists = get("https://api.deezer.com/chart/0/artists", params={"limit": 3})["data"]
    show({"top_tracks": [{"position": t["position"], "title": t["title"], "artist": t["artist"]["name"],
                          "duration_s": t["duration"], "explicit": t["explicit_lyrics"]} for t in tracks],
          "top_artists": [a["name"] for a in artists]})


if __name__ == "__main__":
    main()
