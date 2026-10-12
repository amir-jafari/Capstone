META = dict(
    name="iTunes Search API (Apple music, podcasts, apps, books)",
    domain="Culture, Books & Entertainment",
    docs="https://performance-partners.apple.com/search-api",
    auth="None",
    signup="No registration. About 20 calls/minute. Covers the Apple Music, Podcasts, App Store, Books and TV catalogs.",
    provides=["Search songs, albums, artists, podcasts, apps, audiobooks, movies, TV episodes by keyword",
              "Price, release date, genre, track count, artwork URLs, preview clips; podcast feed URLs",
              "Lookup by iTunes ID, UPC, ISBN or AMG ID; country-specific storefronts"],
    ideas="Podcast landscape analysis (via feed URLs), app-store category studies, music release timing.",
    endpoint="GET https://itunes.apple.com/{search | lookup}",
    params=[
        ("term", "Search text (+ for spaces)", "true crime"),
        ("media / entity", "music | podcast | software | ebook | movie | tvShow ...; entity narrows (album, musicArtist, podcast)", "media=podcast"),
        ("country", "Two-letter storefront", "us"),
        ("attribute", "Field to search: artistTerm, albumTerm, genreIndex, ...", "artistTerm"),
        ("limit", "Max 200", "5"),
        ("lookup?id= / ?upc= / ?isbn=", "Look up by ID; add entity=album to list an artist's albums", "lookup?id=909253&entity=album"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://itunes.apple.com/search", params={"term": "data science", "media": "podcast", "country": "us",
                                                        "limit": 5})
    show({"results": d["resultCount"], "podcasts": [
        {"name": p["collectionName"], "publisher": p["artistName"], "episodes": p.get("trackCount"),
         "genres": p.get("genres", [])[:2], "last_release": (p.get("releaseDate") or "")[:10], "feed": p.get("feedUrl")}
        for p in d["results"]]})


if __name__ == "__main__":
    main()
