META = dict(
    name="USAspending.gov",
    domain="Political Science & Government",
    docs="https://api.usaspending.gov/docs/endpoints",
    auth="None",
    signup="No registration.",
    provides=["Every federal contract, grant, loan award since FY2001: recipient, amount, agency, location, NAICS",
              "Spending by agency, state, congressional district, budget function"],
    ideas="Pork-barrel politics, federal spending vs. district partisanship, contractor concentration.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import requests

from _common import USER_AGENT, show


def main():
    r = requests.post("https://api.usaspending.gov/api/v2/search/spending_by_category/awarding_agency/", json={
        "filters": {"time_period": [{"start_date": "2024-10-01", "end_date": "2025-09-30"}]}, "limit": 5,
    }, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    show([{"agency": x["name"], "obligated_usd": round(x["amount"])} for x in r.json()["results"]])


if __name__ == "__main__":
    main()
