META = dict(
    name="CMS Medicare Part D Prescribers (data.cms.gov)",
    domain="Health & Social Science",
    docs="https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers",
    auth="None",
    signup="No registration. Each yearly release has its own dataset UUID (listed in https://data.cms.gov/data.json). "
           "The same API serves other data.cms.gov datasets (physician services, hospital costs, ...).",
    provides=["Every prescriber's Medicare Part D prescriptions by drug: claims, 30-day fills, total cost, beneficiaries",
              "Prescriber specialty and location; brand vs. generic name", "One row per prescriber x drug (25M+ rows per year)"],
    ideas="Opioid and antibiotic prescribing patterns, brand-name prescribing and pharma payments (Open Payments), cost outliers.",
    endpoint="GET https://data.cms.gov/data-api/v1/dataset/{uuid}/data",
    params=[
        ("uuid", "Dataset release ID from data.json (2024 release shown)", "9552739e-3d05-4c1b-8eff-ecabf391e2e5"),
        ("filter[column]", "Exact-match filter on any column", "filter[Prscrbr_State_Abrvtn]=DC"),
        ("keyword", "Full-text keyword across columns", "semaglutide"),
        ("column", "Comma list of columns to return", "Prscrbr_Last_Org_Name,Gnrc_Name,Tot_Clms,Tot_Drug_Cst"),
        ("size / offset", "Max 5,000 per call", "5"),
        ("/data-api/v1/dataset/{uuid}/data/stats", "Row count for a query", "—"),
    ],
)
from collections import defaultdict

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

URL = "https://data.cms.gov/data-api/v1/dataset/9552739e-3d05-4c1b-8eff-ecabf391e2e5/data"


def main():
    rows = get(URL, params={"filter[Prscrbr_State_Abrvtn]": "DC", "filter[Gnrc_Name]": "Semaglutide", "size": 5000})
    by_type = defaultdict(lambda: {"prescribers": 0, "claims": 0})
    for r in rows:
        t = by_type[r["Prscrbr_Type"]]
        t["prescribers"] += 1
        t["claims"] += int(r["Tot_Clms"])
    show({"drug": "semaglutide, DC prescribers", "prescriber_rows": len(rows),
          "total_cost_usd": round(sum(float(r["Tot_Drug_Cst"]) for r in rows)),
          "by_prescriber_type": sorted(({"type": k, **v} for k, v in by_type.items()), key=lambda x: -x["claims"])})


if __name__ == "__main__":
    main()
