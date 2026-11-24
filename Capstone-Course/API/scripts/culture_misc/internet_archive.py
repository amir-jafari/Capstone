META = dict(
    name="Internet Archive (search & metadata)",
    domain="Culture, Books & Entertainment",
    docs="https://archive.org/developers/",
    auth="None for reading",
    signup="No registration for search, metadata and downloads. Python package: `pip install internetarchive`.",
    provides=["40M+ items: books, films, TV news, audio (Grateful Dead, 78rpm records), software, web collections",
              "Item metadata: title, creator, date, collection, subjects, licence, download and view counts",
              "File lists and direct downloads for every item; the Wayback Machine has its own APIs"],
    ideas="Public-domain film and audio datasets, TV-news transcript analysis, digital-preservation studies.",
    endpoint="GET https://archive.org/advancedsearch.php  ·  GET https://archive.org/metadata/{identifier}",
    params=[
        ("q", "Lucene query over fields: collection:, mediatype:, subject:, year:, creator:", "collection:feature_films AND year:[1920 TO 1959]"),
        ("fl[]", "Fields to return (repeatable)", "identifier, title, year, downloads"),
        ("sort[]", "Sort field and direction", "downloads desc"),
        ("rows / page", "Results per page", "5"),
        ("output", "json | csv | xml", "json"),
        ("/metadata/{identifier}", "Full metadata and file list for one item", "/metadata/night_of_the_living_dead"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://archive.org/advancedsearch.php", params=[
        ("q", 'collection:feature_films AND subject:"film noir"'), ("fl[]", "identifier"), ("fl[]", "title"),
        ("fl[]", "year"), ("fl[]", "downloads"), ("sort[]", "downloads desc"), ("rows", 5), ("output", "json")])["response"]
    show({"film_noir_feature_films": d["numFound"], "most_downloaded": [
        {k: x.get(k) for k in ("title", "year", "downloads")} | {"url": f'https://archive.org/details/{x["identifier"]}'}
        for x in d["docs"]]})


if __name__ == "__main__":
    main()
