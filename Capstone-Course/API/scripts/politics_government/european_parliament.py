META = dict(
    name="European Parliament Open Data API",
    domain="Political Science & Government",
    docs="https://data.europarl.europa.eu/en/developer-corner/opendata-api",
    auth="None",
    signup="No registration. JSON-LD responses (linked data); add format=application/ld+json.",
    provides=["Members (MEPs): country, political group, national party, mandates",
              "Plenary documents, adopted texts, votes and roll calls, parliamentary questions", "Meetings, committees, events"],
    ideas="Comparative legislative behavior (EU vs. US Congress), party-group cohesion, cross-national coalitions.",
    endpoint="GET https://data.europarl.europa.eu/api/v2/{resource}",
    params=[
        ("resource", "meps | meps/show-current | meetings | plenary-documents | adopted-texts | parliamentary-questions | ...", "meps/show-current"),
        ("format", "application/ld+json | application/rdf+xml | text/turtle", "application/ld+json"),
        ("offset / limit", "Paging", "0 / 500"),
        ("country-of-representation", "Filter MEPs by country code (on meps)", "DE"),
        ("year", "Filter documents/meetings by year", "2026"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    meps = get("https://data.europarl.europa.eu/api/v2/meps/show-current",
               params={"format": "application/ld+json", "offset": 0, "limit": 1000})["data"]
    show({"current_MEPs": len(meps),
          "seats_by_group": Counter(m.get("api:political-group") for m in meps).most_common(),
          "sample": [{k: m.get(k) for k in ("label", "api:country-of-representation", "api:political-group")} for m in meps[:3]]},
         max_items=10)


if __name__ == "__main__":
    main()
