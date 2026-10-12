META = dict(
    name="Grants.gov Search API",
    domain="Political Science & Government",
    docs="https://www.grants.gov/api/common/search2",
    auth="None",
    signup="No registration for search (the applicant-side System-to-System API needs registration).",
    provides=["Every federal funding opportunity: title, number, agency, open/close dates, status, CFDA/assistance listing",
              "Filters by eligibility, funding instrument and category", "Opportunity details via fetchOpportunity"],
    ideas="Where federal research money goes by agency/topic over time; matching grants to university strengths.",
    endpoint="POST https://api.grants.gov/v1/api/search2",
    params=[
        ("keyword", "Search text", "data science"),
        ("oppStatuses", "forecasted | posted | closed | archived (pipe-separated)", "forecasted|posted"),
        ("agencies", "Agency codes (pipe-separated)", "NSF|HHS-NIH11"),
        ("fundingCategories / eligibilities / fundingInstruments", "Category codes (e.g. ST science & technology), eligibility codes, G grant / CA cooperative agreement", "ST"),
        ("rows / startRecordNum", "Page size and offset", "5 / 0"),
        ("/v1/api/fetchOpportunity", "Sibling endpoint: full details for one opportunity (POST {opportunityId})", "{\"opportunityId\": 353475}"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show


def main():
    r = requests.post("https://api.grants.gov/v1/api/search2", timeout=60, headers={"User-Agent": USER_AGENT},
                      json={"keyword": "data science", "oppStatuses": "forecasted|posted", "rows": 5})
    r.raise_for_status()
    d = r.json()["data"]
    show({"open_or_forecasted": d["hitCount"], "opportunities": [
        {k: o.get(k) for k in ("number", "title", "agencyCode", "openDate", "closeDate", "oppStatus")} for o in d["oppHits"]]})


if __name__ == "__main__":
    main()
