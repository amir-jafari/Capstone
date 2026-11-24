META = dict(
    name="Lever Postings API (company career pages)",
    domain="Jobs & Labor",
    docs="https://github.com/lever/postings-api",
    auth="None",
    signup="No registration. Companies using Lever (Palantir, Netflix, ...) publish postings at jobs.lever.co/{company}; "
           "the same slug works in the API.",
    provides=["Open postings: title, team, department, location, commitment (full-time/intern), workplace type",
              "Plain-text and HTML descriptions, requirement lists, salary ranges where published", "Apply and hosted URLs"],
    ideas="Comparing job requirements across companies, salary-transparency analysis, internship pipelines.",
    endpoint="GET https://api.lever.co/v0/postings/{company}",
    params=[
        ("mode", "json (default html)", "json"),
        ("team / department / location / commitment", "Filters (exact match)", "commitment=Internship"),
        ("group", "Group results by team | department | location | commitment", "team"),
        ("skip / limit", "Paging", "0 / 50"),
        ("/v0/postings/{company}/{postingId}", "One posting", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    posts = get("https://api.lever.co/v0/postings/palantir", params={"mode": "json"})
    show({"company": "Palantir", "open_postings": len(posts),
          "by_team": Counter(p["categories"].get("team") for p in posts).most_common(5),
          "by_commitment": Counter(p["categories"].get("commitment") for p in posts).most_common(),
          "by_workplace": Counter(p.get("workplaceType") for p in posts).most_common(),
          "examples": [{"title": p["text"], "location": p["categories"].get("location")} for p in posts[:3]]})


if __name__ == "__main__":
    main()
