META = dict(
    name="Art Institute of Chicago API",
    domain="Culture, Books & Entertainment",
    docs="https://api.artic.edu/docs/",
    auth="None",
    signup="No registration. 60 requests/minute; use the fields= parameter to keep responses small. CC0 for public-domain "
           "works; high-resolution images via IIIF.",
    provides=["120k+ artworks: title, artist, date, medium, dimensions, place of origin, department, style, subject terms",
              "Public-domain flag and IIIF image URLs, color data, exhibition history", "Artists, exhibitions, tours, publications"],
    ideas="Computer vision on museum images, color and style evolution over time, provenance and acquisition patterns.",
    endpoint="GET https://api.artic.edu/api/v1/{artworks | artworks/search | artists | exhibitions}",
    params=[
        ("q", "Full-text search", "monet"),
        ("query (search)", "Elasticsearch query DSL as JSON for precise filters", "{\"term\": {\"is_public_domain\": true}}"),
        ("fields", "Fields to return", "id,title,artist_title,date_display,image_id,is_public_domain"),
        ("limit / page", "Max 100", "5"),
        ("IIIF image URL", "{config.iiif_url}/{image_id}/full/843,/0/default.jpg", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.artic.edu/api/v1/artworks/search", params={
        "q": "monet", "limit": 5,
        "fields": "id,title,artist_title,date_display,medium_display,is_public_domain,image_id"})
    iiif = d["config"]["iiif_url"]
    show({"matches": d["pagination"]["total"], "artworks": [
        {"title": a["title"], "artist": a["artist_title"], "date": a["date_display"], "public_domain": a["is_public_domain"],
         "image": f'{iiif}/{a["image_id"]}/full/843,/0/default.jpg' if a.get("image_id") else None} for a in d["data"]]})


if __name__ == "__main__":
    main()
