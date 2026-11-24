META = dict(
    name="NIH RePORTER API",
    domain="Science & Research",
    docs="https://api.reporter.nih.gov/",
    auth="None",
    signup="No registration. POST JSON criteria. Max 1 request/second; max 500 records per request and 15,000 per "
           "query (narrow by fiscal year for more).",
    provides=["Every NIH-funded project: title, abstract, PI, organization, award amount, institute (NCI, NIMH, ...), terms",
              "Fiscal year, activity code (R01, K99, ...), study section, spending categories",
              "Linked publications (publications/search endpoint)"],
    ideas="Biomedical funding by disease area, success predictors for grants, linking grants to papers (PubMed).",
    endpoint="POST https://api.reporter.nih.gov/v2/projects/search",
    params=[
        ("criteria.fiscal_years", "List of fiscal years", "[2025]"),
        ("criteria.org_names / org_states", "Organization names (upper case) / states", "[\"GEORGE WASHINGTON UNIVERSITY\"]"),
        ("criteria.advanced_text_search", "{operator, search_field, search_text}", "{\"search_field\": \"projecttitle,terms\", \"search_text\": \"machine learning\"}"),
        ("criteria.agencies / activity_codes", "Institute codes / grant types", "[\"NIMH\"] / [\"R01\"]"),
        ("include_fields", "Fields to return", "[\"ProjectTitle\", \"AwardAmount\", \"Organization\"]"),
        ("offset / limit / sort_field / sort_order", "Paging and sorting (limit max 500)", "0 / 5 / award_amount / desc"),
        ("/v2/publications/search", "Sibling endpoint: publications linked to projects", "—"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show


def main():
    body = {"criteria": {"fiscal_years": [2025], "org_names": ["GEORGE WASHINGTON UNIVERSITY"]},
            "include_fields": ["ProjectTitle", "AwardAmount", "AgencyIcAdmin", "ActivityCode", "PrincipalInvestigators"],
            "offset": 0, "limit": 5, "sort_field": "award_amount", "sort_order": "desc"}
    r = requests.post("https://api.reporter.nih.gov/v2/projects/search", json=body, timeout=60,
                      headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    d = r.json()
    show({"gwu_nih_projects_fy2025": d["meta"]["total"], "largest": [
        {"title": p["project_title"], "amount_usd": p["award_amount"], "institute": (p.get("agency_ic_admin") or {}).get("abbreviation"),
         "activity": p["activity_code"], "pi": (p.get("principal_investigators") or [{}])[0].get("full_name")}
        for p in d["results"]]})


if __name__ == "__main__":
    main()
