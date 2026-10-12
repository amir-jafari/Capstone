META = dict(
    name="GovInfo API (U.S. Government Publishing Office)",
    domain="Political Science & Government",
    docs="https://api.govinfo.gov/docs/ · https://github.com/usgpo/api",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use your api.data.gov key (https://api.data.gov/signup/). 36,000 req/hour with a personal key.",
    provides=["Official full text + metadata of 40+ collections: bills (BILLS), Congressional Record (CREC), US Code (USCODE)",
              "Code of Federal Regulations (CFR), Federal Register (FR), federal court opinions (USCOURTS), GAO reports, budget",
              "Every document as PDF, HTML, XML and MODS metadata; search across collections"],
    ideas="Text-as-data on bills and floor speeches, tracking legislative language diffusion, court-opinion corpora.",
    endpoint="POST https://api.govinfo.gov/search  ·  GET https://api.govinfo.gov/collections/{collection}/{since}",
    params=[
        ("query (POST body)", "Search with field operators: collection:, congress:, docClass:, publishdate:range(...)", "collection:USCOURTS AND \"qualified immunity\""),
        ("pageSize / offsetMark (POST body)", "Page size (max 1,000); offsetMark=* then the returned mark for paging", "5 / *"),
        ("sorts (POST body)", "List of {field, sortOrder}", "[{\"field\": \"publishdate\", \"sortOrder\": \"DESC\"}]"),
        ("/collections", "List collections with document counts", "—"),
        ("/collections/{code}/{lastModifiedStartDate}", "Packages added/changed since a date (pageSize, offsetMark)", "/collections/BILLS/2026-10-01T00:00:00Z"),
        ("/packages/{packageId}/summary", "One document's metadata and download links (htm, xml, pdf)", "/packages/BILLS-119hr1enr/summary"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import os

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show


def main():
    r = requests.post("https://api.govinfo.gov/search", timeout=60, headers={"User-Agent": USER_AGENT},
                      params={"api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")}, json={
                          "query": 'collection:USCOURTS AND "qualified immunity"', "pageSize": 5, "offsetMark": "*",
                          "sorts": [{"field": "publishdate", "sortOrder": "DESC"}]})
    r.raise_for_status()
    d = r.json()
    show({"matching_court_documents": d["count"], "latest": [
        {"title": x["title"], "court": x["governmentAuthor"][0], "dateIssued": x["dateIssued"], "packageId": x["packageId"]}
        for x in d["results"]]})


if __name__ == "__main__":
    main()
