META = dict(
    name="Gutendex (Project Gutenberg books)",
    domain="Culture, Books & Entertainment",
    docs="https://gutendex.com/",
    auth="None",
    signup="No registration (uncached queries can take 1-2 minutes, so use long timeouts). JSON index of Project Gutenberg's 75,000+ free public-domain e-books; full texts download "
           "as plain text, EPUB or HTML from the formats links.",
    provides=["Books: title, authors (with birth/death years), subjects, bookshelves, languages, download counts",
              "Direct links to full text in multiple formats", "Search, topic and language filters, popularity sort"],
    ideas="Classic-literature NLP corpora, stylometry and authorship attribution, sentiment arcs of novels.",
    endpoint="GET https://gutendex.com/books/",
    params=[
        ("search", "Words in title or author name", "austen"),
        ("topic", "Words in subjects or bookshelves", "detective"),
        ("languages", "Comma-separated codes", "en,fr"),
        ("author_year_start / author_year_end", "Authors alive in a year range", "1800 / 1899"),
        ("sort", "popular (default) | ascending | descending (by ID)", "popular"),
        ("ids", "Specific Gutenberg IDs", "1342,84"),
        ("page", "32 results per page", "1"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://gutendex.com/books/", params={"topic": "detective", "languages": "en"},
            timeout=120)  # uncached queries can take 1-2 minutes on this free server
    show({"matching_books": d["count"], "most_downloaded": [
        {"id": b["id"], "title": b["title"], "author": (b["authors"] or [{}])[0].get("name"),
         "downloads": b["download_count"], "plain_text": b["formats"].get("text/plain; charset=us-ascii")
                                                         or b["formats"].get("text/plain; charset=utf-8")}
        for b in d["results"][:5]]})


if __name__ == "__main__":
    main()
