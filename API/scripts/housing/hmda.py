META = dict(
    name="HMDA Data Browser (mortgage applications)",
    domain="Housing & Real Estate",
    docs="https://ffiec.cfpb.gov/documentation/api/data-browser/",
    auth="None",
    signup="No registration. Home Mortgage Disclosure Act data: nearly every US mortgage application since 2018, "
           "published each year by the CFPB/FFIEC.",
    provides=["Loan-level records: action taken (originated/denied), loan amount, purpose, type, interest rate, property value",
              "Applicant race, ethnicity, sex, age, income; denial reasons; lender (LEI)", "Census tract of the property"],
    ideas="Fair-lending / mortgage-denial disparities, credit access by neighborhood, lender behavior over the rate cycle.",
    endpoint="GET https://ffiec.cfpb.gov/v2/data-browser-api/view/{aggregations | csv}",
    params=[
        ("years", "Data year(s)", "2023"),
        ("states / msamds / counties / leis", "Geography or lender filter (one required); msamds uses metro-division codes", "states=DC, msamds=47894"),
        ("actions_taken", "1 originated, 2 approved not accepted, 3 denied, 4 withdrawn, ... (comma list)", "1,3"),
        ("races / ethnicities / sexes", "Applicant demographics (comma lists; each value returns its own row). Aggregations accept at most 2 of these filter variables per query", "White,Black or African American,Asian"),
        ("loan_purposes / loan_types", "1 purchase, 2 improvement, 31 refinance, 32 cash-out; 1 conventional, 2 FHA, 3 VA, 4 USDA", "1"),
        ("/view/csv", "Same filters, redirects to a CSV file of the matching loan-level rows", "/view/csv?years=2023&states=DC"),
    ],
)
from collections import defaultdict

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations", params={
        "years": 2023, "states": "DC", "actions_taken": "1,3",
        "races": "White,Black or African American,Asian"})["aggregations"]
    by_race = defaultdict(dict)
    for row in d:  # one row per (action_taken, race)
        by_race[row["races"]]["originated" if row["actions_taken"] == "1" else "denied"] = row["count"]
    # denial rate among decided applications only: denied / (denied + originated)
    show([{"race": r, **c, "denial_rate": round(c["denied"] / (c["denied"] + c["originated"]), 3)}
          for r, c in by_race.items()])


if __name__ == "__main__":
    main()
