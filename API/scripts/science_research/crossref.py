META = dict(
    name="Crossref REST API",
    domain="Science & Research",
    docs="https://api.crossref.org/swagger-ui/index.html",
    auth="None (add `mailto=` for polite pool)",
    signup="No registration.",
    provides=["Metadata for 150M+ DOIs: title, authors, journal, publisher, dates, references, funders, licenses"],
    ideas="Bibliometrics, funder-impact analysis, DOI metadata enrichment.",
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
