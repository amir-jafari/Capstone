META = dict(
    name="NASA Image and Video Library",
    domain="Science & Research",
    docs="https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf",
    auth="None",
    signup="No registration (separate from api.nasa.gov - no DEMO_KEY limits). Most NASA media are not copyrighted, "
           "but check the usage guidelines for logos and identifiable people.",
    provides=["140k+ NASA images, videos and audio: title, description, keywords, center, photographer, date",
              "Asset manifests with original-resolution files and captions", "Search by text, center, year range, media type"],
    ideas="Space-image datasets for computer vision, captioning models, mission-history timelines.",
    endpoint="GET https://images-api.nasa.gov/{search | asset/{nasa_id} | metadata/{nasa_id}}",
    params=[
        ("q", "Free-text search", "james webb"),
        ("media_type", "image | video | audio (comma list)", "image"),
        ("center / keywords / photographer", "Field filters", "center=GSFC"),
        ("year_start / year_end", "Date range", "2022 / 2026"),
        ("page / page_size", "Paging (page_size max 100)", "1 / 5"),
        ("/asset/{nasa_id}", "All file URLs (orig, large, medium, thumb) for one item", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://images-api.nasa.gov/search", params={"q": "james webb deep field", "media_type": "image",
                                                          "year_start": 2022, "page_size": 5})["collection"]
    show({"hits": d["metadata"]["total_hits"], "images": [
        {"title": i["data"][0]["title"], "date": i["data"][0]["date_created"][:10], "center": i["data"][0].get("center"),
         "nasa_id": i["data"][0]["nasa_id"], "thumbnail": i["links"][0]["href"]} for i in d["items"]]})


if __name__ == "__main__":
    main()
