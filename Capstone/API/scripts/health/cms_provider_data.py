META = dict(
    name="CMS Provider Data Catalog (Care Compare)",
    domain="Health & Social Science",
    docs="https://data.cms.gov/provider-data/docs",
    auth="None",
    signup="No registration. The data behind Medicare's Care Compare sites: hospitals, nursing homes, dialysis, home health, "
           "physicians. Dataset IDs are on each dataset page.",
    provides=["Hospital general information and overall star ratings, readmissions, mortality, patient-experience (HCAHPS)",
              "Nursing-home ratings, staffing and penalties; dialysis and home-health quality",
              "SQL-like filtering, sorting and paging through the DKAN datastore API"],
    ideas="Hospital quality vs. ownership or neighborhood, readmission prediction, nursing-home staffing and outcomes.",
    endpoint="GET https://data.cms.gov/provider-data/api/1/datastore/query/{datasetId}/0",
    params=[
        ("datasetId", "Dataset ID: xubh-q36u Hospital General Information, ...", "xubh-q36u"),
        ("conditions[0][property] / [value] / [operator]", "Filters (=, <>, >, <, LIKE, IN)", "state = DC"),
        ("sorts[0][property] / [order]", "Sort", "hospital_overall_rating / desc"),
        ("limit / offset", "Max 1,500 per call", "5"),
        ("count / results", "true|false: include total count / rows", "true"),
        ("/api/1/metastore/schemas/dataset/items", "Sibling endpoint: catalog of all datasets", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0", params={
        "conditions[0][property]": "state", "conditions[0][value]": "MD", "conditions[0][operator]": "=", "limit": 200})
    rows = d["results"]
    show({"maryland_hospitals": d["count"],
          "overall_star_ratings": dict(sorted(Counter(r["hospital_overall_rating"] for r in rows).items())),
          "five_star": [{k: r[k] for k in ("facility_name", "citytown", "hospital_type", "hospital_ownership")}
                        for r in rows if r["hospital_overall_rating"] == "5"][:3]})


if __name__ == "__main__":
    main()
