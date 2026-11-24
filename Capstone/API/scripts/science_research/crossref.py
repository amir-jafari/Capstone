META = dict(
    name="Crossref REST API",
    domain="Science & Research",
    docs="https://api.crossref.org/swagger-ui/index.html",
    auth="None (add `mailto=` for polite pool)",
    signup="No registration.",
    provides=["Metadata for 150M+ DOIs: title, authors, journal, publisher, dates, references, funders, licenses"],
    ideas="Bibliometrics, funder-impact analysis, DOI metadata enrichment.",
    endpoint="GET https://api.crossref.org/works",
    params=[
        ("query.bibliographic / query.author / query", "Citation-style, author, or general search", "data science education"),
        ("filter", "Comma list: from-pub-date, until-pub-date, type, has-funder, issn, ...", "from-pub-date:2020-01-01,type:journal-article"),
        ("sort / order", "is-referenced-by-count | published | relevance; asc | desc", "is-referenced-by-count / desc"),
        ("rows / offset", "Max 1,000 per page", "3"),
        ("cursor", "Deep paging (start with *)", "*"),
        ("select", "Fields to return", "DOI,title,author"),
        ("mailto", "Your email → polite pool", "you@gwu.edu"),
        ("/works/{doi}", "Sibling endpoint: one DOI's metadata", "/works/10.1038/nature14539"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.crossref.org/works", params={
        "query.bibliographic": "data science education", "rows": 3, "sort": "is-referenced-by-count",
        "order": "desc", "mailto": "your_email@gwu.edu"})["message"]
    show([{"DOI": w["DOI"], "title": w["title"][0], "journal": (w.get("container-title") or [None])[0],
           "year": w["issued"]["date-parts"][0][0], "cited_by": w["is-referenced-by-count"]} for w in d["items"]])


if __name__ == "__main__":
    main()
