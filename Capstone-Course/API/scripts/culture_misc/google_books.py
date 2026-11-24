META = dict(
    name="Google Books API",
    domain="Culture, Books & Entertainment",
    docs="https://developers.google.com/books/docs/v1/using",
    auth="API key (query `key`) - effectively required",
    env="GOOGLE_BOOKS_API_KEY",
    signup="In https://console.cloud.google.com create a project, enable 'Books API', and create an API key. Keyless calls "
           "share a global quota that is often exhausted (HTTP 429). 1,000 requests/day per key by default.",
    provides=["Tens of millions of books: title, authors, publisher, published date, ISBNs, page count, categories",
              "Descriptions, average rating, language, cover thumbnails, preview/ebook availability and price"],
    ideas="Publishing trends by subject, metadata enrichment of book datasets via ISBN, description-based recommendation.",
    endpoint="GET https://www.googleapis.com/books/v1/volumes",
    params=[
        ("q", "Search with keywords: intitle:, inauthor:, subject:, isbn:, inpublisher:", "subject:statistics"),
        ("orderBy", "relevance | newest", "newest"),
        ("printType / filter", "all | books | magazines; partial | full | free-ebooks | paid-ebooks | ebooks", "books"),
        ("langRestrict", "Two-letter language code", "en"),
        ("maxResults / startIndex", "Max 40 per page", "5 / 0"),
        ("key", "Your API key", "<GOOGLE_BOOKS_API_KEY>"),
    ],
    sample="""[
  {"title": "An Introduction to Statistical Learning", "authors": ["Gareth James", "Daniela Witten", "Trevor Hastie", "Robert Tibshirani"],
   "publishedDate": "2023-07-01", "pageCount": 607, "categories": ["Mathematics"]}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://www.googleapis.com/books/v1/volumes", params={
        "q": "subject:statistics", "orderBy": "newest", "printType": "books", "maxResults": 5,
        "key": key("GOOGLE_BOOKS_API_KEY")})
    show([{k: b["volumeInfo"].get(k) for k in ("title", "authors", "publishedDate", "pageCount", "categories")}
          for b in d.get("items", [])])


if __name__ == "__main__":
    main()
