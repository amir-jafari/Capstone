META = dict(
    name="U.S. Department of Justice News API",
    domain="Law & Justice",
    docs="https://www.justice.gov/developer",
    auth="None",
    signup="No registration. Covers DOJ press releases (incl. all 94 U.S. Attorney's Offices) and blog posts.",
    provides=["270,000+ press releases: title, date, full HTML body, DOJ components (office/division), topics",
              "Prosecutions, sentencings, settlements and indictments across fraud, cybercrime, civil rights, environment..."],
    ideas="Enforcement-priority shifts across administrations, NLP extraction of charges and sentences, topic trends.",
    endpoint="GET https://www.justice.gov/api/v1/{press_releases | blog_entries}.json",
    params=[
        ("pagesize / page", "Page size and page number (0-based)", "5 / 0"),
        ("sort / direction", "Sort field and order", "date / DESC"),
        ("parameters[title]", "Filter by words in the title (other fields: parameters[{field}])", "fraud"),
        ("fields", "date = Unix timestamp; component[].name = DOJ office; topic[].name = subject", "—"),
    ],
)
import datetime as dt
import html
import re

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.justice.gov/api/v1/press_releases.json", params={
        "pagesize": 5, "sort": "date", "direction": "DESC", "parameters[title]": "artificial intelligence"})
    show({"matching_releases": d["metadata"]["resultset"]["count"], "latest": [
        {"date": dt.date.fromtimestamp(int(r["date"])).isoformat(), "title": r["title"],
         "offices": [c["name"] for c in r.get("component") or []][:2],
         "topics": [t.get("name") for t in r.get("topic") or []],
         "teaser": html.unescape(re.sub("<[^>]+>", "", r.get("teaser") or "")).strip()}
        for r in d["results"]]})


if __name__ == "__main__":
    main()
