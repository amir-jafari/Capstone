META = dict(
    name="OpenFEC (Federal Election Commission)",
    domain="Political Science & Government",
    docs="https://api.open.fec.gov/developers/",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use the same api.data.gov key (https://api.data.gov/signup/). 1,000 req/hour.",
    provides=["Candidates & committees, total receipts / disbursements / cash on hand",
              "Itemized individual contributions (Schedule A): donor name, employer, occupation, ZIP, amount",
              "Independent expenditures, PAC spending, filings"],
    ideas="Campaign-finance networks, donor geography, money vs. electoral outcomes.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    d = get("https://api.open.fec.gov/v1/candidates/totals/", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "election_year": 2024, "office": "P",
        "sort": "-receipts", "per_page": 5, "is_active_candidate": True})
    show([{"name": c["name"], "party": c["party"], "receipts": c["receipts"],
           "disbursements": c["disbursements"]} for c in d["results"]])


if __name__ == "__main__":
    main()
