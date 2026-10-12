META = dict(
    name="ERIC (Education Resources Information Center)",
    domain="Education",
    docs="https://eric.ed.gov/?api",
    auth="None",
    signup="No registration. Run by the Institute of Education Sciences (IES).",
    provides=["1.9M+ education research records: title, authors, abstract, descriptors (subject terms), publication type",
              "Peer-review flag, full-text availability (ERIC-hosted PDFs), education level, audience, source journal"],
    ideas="Topic modeling of education research trends, systematic-review automation, citation-free literature mapping.",
    endpoint="GET https://api.ies.ed.gov/eric/",
    params=[
        ("search", "Query; supports field syntax (title:, author:, subject:, source:) and AND / OR", "subject:\"mathematics anxiety\""),
        ("fields", "Comma-separated fields to return", "id,title,author,publicationdateyear,peerreviewed"),
        ("format", "json | xml | csv", "json"),
        ("rows / start", "Max 2,000 per request; offset for paging", "5"),
        ("Filter terms in search", "peerreviewed:T, e_fulltextauth:1, publicationdateyear:2024, educationlevel:\"Higher Education\"", "peerreviewed:T"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.ies.ed.gov/eric/", params={
        "search": 'subject:"artificial intelligence" AND peerreviewed:T AND publicationdateyear:2025',
        "fields": "id,title,author,source,publicationdateyear", "format": "json", "rows": 5})["response"]
    show({"numFound": d["numFound"], "docs": d["docs"]})


if __name__ == "__main__":
    main()
