META = dict(
    name="MusicBrainz (open music encyclopedia)",
    domain="Culture, Books & Entertainment",
    docs="https://musicbrainz.org/doc/MusicBrainz_API",
    auth="None (descriptive User-Agent required)",
    signup="No registration. Max 1 request/second; send a User-Agent like 'AppName/1.0 (contact@example.com)'. "
           "Full database dumps are available for bulk analysis. CC0 core data.",
    provides=["Artists, release groups (albums), releases, recordings, works, labels with stable MBIDs",
              "Relationships (members, producers, covers, samples), tags/genres, release dates and countries",
              "Links to Wikidata, Discogs, Spotify and cover art (Cover Art Archive)"],
    ideas="Music-history networks (collaborations), genre evolution, joining streaming data via MBIDs.",
    endpoint="GET https://musicbrainz.org/ws/2/{entity}/[{mbid}]",
    params=[
        ("entity", "artist | release-group | release | recording | work | label | area | event", "artist"),
        ("query (search)", "Lucene syntax with fields: artist:, country:, tag:, type:, date:", "artist:radiohead"),
        ("artist= / release= (browse)", "List entities linked to an MBID", "release-group?artist={mbid}&type=album"),
        ("inc (lookup)", "Include related data: aliases, tags, genres, recordings, url-rels, artist-rels", "inc=genres+url-rels"),
        ("fmt", "json | xml", "json"),
        ("limit / offset", "Max 100", "5"),
    ],
)
import time

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://musicbrainz.org/ws/2"


def main():
    artist = get(f"{BASE}/artist/", params={"query": "artist:radiohead", "fmt": "json", "limit": 1})["artists"][0]
    time.sleep(1.1)  # MusicBrainz allows 1 request/second
    albums = get(f"{BASE}/release-group", params={"artist": artist["id"], "type": "album", "fmt": "json", "limit": 100})
    albums_only = sorted((rg for rg in albums["release-groups"] if not rg["secondary-types"]),  # no live/compilation
                    key=lambda rg: rg.get("first-release-date") or "")
    show({"artist": artist["name"], "mbid": artist["id"], "country": artist.get("country"),
          "tags": [t["name"] for t in sorted(artist.get("tags", []), key=lambda t: -t["count"])][:5],
          "albums": [f'{rg.get("first-release-date", "")[:4]} {rg["title"]}' for rg in albums_only]}, max_items=10)


if __name__ == "__main__":
    main()
