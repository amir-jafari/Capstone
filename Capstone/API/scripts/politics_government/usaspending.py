META = dict(
    name="USAspending.gov",
    domain="Political Science & Government",
    docs="https://api.usaspending.gov/docs/endpoints",
    auth="None",
    signup="No registration.",
    provides=["Every federal contract, grant, loan award since FY2001: recipient, amount, agency, location, NAICS",
              "Spending by agency, state, congressional district, budget function"],
    ideas="Pork-barrel politics, federal spending vs. district partisanship, contractor concentration.",
    endpoint="POST https://api.usaspending.gov/api/v2/search/spending_by_category/{category}/",
    params=[
        ("category (path)", "awarding_agency | awarding_subagency | recipient | naics | psc | cfda | state_territory | county | district | ...", "awarding_agency"),
        ("filters.time_period", "List of {start_date, end_date} (fiscal year starts Oct 1)", "[{\"start_date\": \"2024-10-01\", \"end_date\": \"2025-09-30\"}]"),
        ("filters.award_type_codes", "A-D = contracts, 02-05 = grants, 07/08 = loans", "[\"A\", \"B\", \"C\", \"D\"]"),
        ("filters.agencies", "List of {type, tier, name}", "[{\"type\": \"awarding\", \"tier\": \"toptier\", \"name\": \"Department of Energy\"}]"),
        ("filters.recipient_locations", "List of {country, state, county, district}", "[{\"country\": \"USA\", \"state\": \"VA\"}]"),
        ("limit / page", "Paging", "5"),
        ("/api/v2/search/spending_by_award/", "Sibling endpoint: individual awards (same filters + fields list)", "fields: [\"Award ID\", \"Recipient Name\", \"Award Amount\"]"),
    ],
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
