META = dict(
    name="NPPES NPI Registry (healthcare providers)",
    domain="Health & Social Science",
    docs="https://npiregistry.cms.hhs.gov/api-page",
    auth="None",
    signup="No registration. Every US healthcare provider and organization that bills insurance has a National Provider "
           "Identifier (NPI). Full monthly file available for bulk use.",
    provides=["Provider name, credential, NPI type (individual/organization), taxonomy (specialty), practice addresses",
              "Search by name, specialty, city, state, ZIP"],
    ideas="Provider-supply maps (specialists per capita, health-care deserts), linking NPIs to Medicare claims data.",
    endpoint="GET https://npiregistry.cms.hhs.gov/api/",
    params=[
        ("version", "API version (required)", "2.1"),
        ("number", "Look up one NPI", "1234567893"),
        ("first_name / last_name / organization_name", "Name search (* wildcard after 2 letters)", "last_name=Smith"),
        ("taxonomy_description", "Specialty text", "cardiology"),
        ("city / state / postal_code", "Location", "Washington / DC"),
        ("enumeration_type", "NPI-1 (individual) | NPI-2 (organization)", "NPI-1"),
        ("limit / skip", "Max 200 per call; skip up to 1,000", "5"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://npiregistry.cms.hhs.gov/api/", params={
        "version": "2.1", "taxonomy_description": "pediatrics", "city": "Washington", "state": "DC",
        "enumeration_type": "NPI-1", "limit": 200})
    specialties = Counter(t["desc"] for r in d["results"] for t in r["taxonomies"] if t["primary"])
    show({"pediatrics_providers_returned": d["result_count"], "primary_specialties": specialties.most_common(5),
          "examples": [{"npi": r["number"], "name": f'{r["basic"].get("first_name", "")} {r["basic"].get("last_name", "")}',
                        "credential": r["basic"].get("credential")} for r in d["results"][:3]]})


if __name__ == "__main__":
    main()
