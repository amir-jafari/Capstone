META = dict(
    name="GLEIF Legal Entity Identifier (LEI) API",
    domain="Finance & Economics",
    docs="https://www.gleif.org/en/lei-data/gleif-api",
    auth="None",
    signup="No registration. The global registry of 2.8M+ Legal Entity Identifiers (required for entities trading in "
           "financial markets), with ownership links. CC0.",
    provides=["Legal name, addresses, jurisdiction, legal form, registration status for each LEI",
              "Direct and ultimate parent relationships (who owns whom)", "Mappings LEI <-> ISIN, BIC, OpenCorporates, S&P/MIC"],
    ideas="Corporate ownership networks, linking datasets across identifiers (SEC, FDIC, trade data), offshore-structure studies.",
    endpoint="GET https://api.gleif.org/api/v1/{lei-records | fuzzycompletions | lei-records/{lei}/...}",
    params=[
        ("filter[entity.legalName]", "Exact legal name (also filter[entity.legalAddress.country], filter[entity.status])", "Apple Inc."),
        ("fuzzycompletions?field=entity.legalName&q=", "Typo-tolerant name search", "q=micro soft"),
        ("lei-records/{lei}/direct-parent · /ultimate-parent · /direct-children", "Ownership relationships (404 = no parent reported)", "—"),
        ("lei-records/{lei}/isins", "ISINs (securities) issued by the entity", "—"),
        ("page[size] / page[number]", "Paging (max 200)", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://api.gleif.org/api/v1"


def main():
    rec = get(f"{BASE}/lei-records", params={"filter[entity.legalName]": "Alphabet Inc.", "page[size]": 1})["data"][0]
    kids = get(f'{BASE}/lei-records/{rec["id"]}/direct-children', params={"page[size]": 5})
    e = rec["attributes"]["entity"]
    show({"lei": rec["id"], "legal_name": e["legalName"]["name"], "jurisdiction": e["jurisdiction"],
          "status": e["status"], "hq_city": e["headquartersAddress"]["city"],
          "direct_children_registered": kids["meta"]["pagination"]["total"],
          "examples": [c["attributes"]["entity"]["legalName"]["name"] for c in kids["data"]]})


if __name__ == "__main__":
    main()
