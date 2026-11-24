META = dict(
    name="Supreme Court Database (SCDB)",
    domain="Law & Justice",
    docs="http://scdb.wustl.edu/documentation.php",
    auth="None (CSV download)",
    signup="No registration; no REST API. Download zipped CSVs from http://scdb.wustl.edu/data.php - a new release "
           "each year (file names carry the release, e.g. 2026_01). Case-centered and justice-centered versions.",
    provides=["Every Supreme Court decision since 1791 coded on 200+ variables: issue area, decision direction "
              "(liberal/conservative), vote split, parties, lower court, precedent alteration",
              "Justice-centered file: how each justice voted and whether they wrote an opinion"],
    ideas="The standard dataset for judicial-politics research: ideal points, court ideology over time, outcome prediction.",
    endpoint="GET http://scdb.wustl.edu/_brickFiles/{release}/SCDB_{release}_{caseCentered | justiceCentered}_Citation.csv.zip",
    params=[
        ("release", "Yearly release ID", "2026_01"),
        ("caseCentered / justiceCentered", "One row per case, or one row per justice vote", "caseCentered"),
        ("Citation / Docket / LegalProvision", "Unit of analysis inside the file name", "Citation"),
        ("key columns", "term, caseName, dateDecision, issueArea, decisionDirection (1 conservative, 2 liberal), majVotes, minVotes, chief", "—"),
    ],
)
import csv
import io
import zipfile
from collections import Counter

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

URL = "http://scdb.wustl.edu/_brickFiles/2026_01/SCDB_2026_01_caseCentered_Citation.csv.zip"


def main():
    z = zipfile.ZipFile(io.BytesIO(requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=120).content))
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1")))
    terms = sorted({r["term"] for r in rows})[-5:]
    out = []
    for t in terms:
        cases = [r for r in rows if r["term"] == t]
        direction = Counter(r["decisionDirection"] for r in cases)
        out.append({"term": t, "cases": len(cases), "liberal_share": round(direction["2"] / (direction["1"] + direction["2"]), 2),
                    "unanimous": sum(r["minVotes"] == "0" for r in cases)})
    show({"cases_in_database": len(rows), "last_5_terms": out})


if __name__ == "__main__":
    main()
