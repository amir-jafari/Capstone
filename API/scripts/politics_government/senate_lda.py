META = dict(
    name="Lobbying Disclosure Act (LDA) API",
    domain="Political Science & Government",
    docs="https://lda.gov/api/redoc/v1/",
    auth="None (optional free key raises limits)",
    env="LDA_API_KEY (optional)",
    signup="Works anonymously at low rates. Register at https://lda.gov/api/register/ for a key "
           "(header `Authorization: Token <key>`). Formerly lda.senate.gov.",
    provides=["Every federal lobbying registration and quarterly report since 1999",
              "Registrant (firm), client, income/expenses, issue codes, specific bills/issues lobbied, agencies contacted",
              "Individual lobbyists (incl. former government positions), political contributions (LD-203)"],
    ideas="Lobbying spend vs. legislative outcomes, revolving-door networks, which industries lobby on AI or climate.",
    endpoint="GET https://lda.gov/api/v1/{resource}/",
    params=[
        ("resource", "filings | contributions | registrants | clients | lobbyists | constants/filing/lobbyingactivityissues", "filings"),
        ("filing_year / filing_period", "Year; first_quarter | second_quarter | third_quarter | fourth_quarter", "2025 / second_quarter"),
        ("filing_specific_lobbying_issues", "Full-text search in the issues/bills lobbied", "artificial intelligence"),
        ("client_name / registrant_name", "Name search", "Microsoft"),
        ("filing_type", "RR registration, Q1-Q4 quarterly reports, ... ", "Q2"),
        ("filing_amount_reported_min / _max", "Reported income (lobbying firms) or expenses (in-house lobbying) range in USD", "100000"),
        ("page / page_size", "Paging (max 25 anonymous)", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://lda.gov/api/v1/filings/", params={
        "filing_year": 2025, "filing_specific_lobbying_issues": "artificial intelligence",
        "filing_amount_reported_min": 1, "page_size": 10})
    unique = {f["filing_uuid"]: f for f in d["results"]}.values()  # the same filing can appear more than once
    show({"filings_mentioning_AI_2025": d["count"], "examples": [
        {"client": f["client"]["name"], "registrant": f["registrant"]["name"], "period": f["filing_period_display"],
         "amount_usd": f["income"] or f["expenses"],  # firms report income, in-house lobbyists report expenses
         "issues": sorted({a["general_issue_code_display"] for a in f["lobbying_activities"]})[:3]}
        for f in list(unique)[:5]]})


if __name__ == "__main__":
    main()
