META = dict(
    name="Open Library (books)",
    domain="Culture, Books & Entertainment",
    docs="https://openlibrary.org/developers/api",
    auth="None",
    signup="No registration.",
    provides=["Book search: title, authors, first publish year, subjects, editions, ISBNs", "Author & cover data"],
    ideas="Cultural analytics, book-recommendation systems, publishing trends.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://openlibrary.org/search.json", params={"q": "data science", "limit": 5,
                                                          "fields": "title,author_name,first_publish_year,edition_count"})
    show({"numFound": d["numFound"], "docs": d["docs"]})


if __name__ == "__main__":
    main()
