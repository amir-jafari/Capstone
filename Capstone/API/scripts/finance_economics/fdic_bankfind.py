META = dict(
    name="FDIC BankFind Suite API",
    domain="Finance & Economics",
    docs="https://api.fdic.gov/banks/docs/",
    auth="None",
    signup="No registration. Every FDIC-insured bank: structure, quarterly financials (Call Reports), branches, failures.",
    provides=["Institutions: name, charter, assets, deposits, location, regulator, active/inactive, history",
              "Quarterly financials (1,000+ Call Report fields), branch locations with deposits (Summary of Deposits)",
              "Bank failures since 1934 with estimated cost to the deposit insurance fund"],
    ideas="Bank-failure prediction, deposit concentration and banking deserts, bank-merger trends.",
    endpoint="GET https://api.fdic.gov/banks/{institutions | financials | locations | failures | history | summary}",
    params=[
        ("filters", "Elasticsearch-style filter string: STALP:DC, ACTIVE:1, ASSET:[1000000 TO *]", "STALP:VA AND ACTIVE:1"),
        ("fields", "Comma list of fields", "NAME,CITY,ASSET,DEP"),
        ("sort_by / sort_order", "Field and ASC | DESC", "ASSET / DESC"),
        ("limit / offset", "Max 10,000", "5"),
        ("format", "json | csv", "json"),
        ("failures?filters=FAILYR:2023", "Bank failures with cost estimates", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://api.fdic.gov/banks"


def main():
    big = get(f"{BASE}/institutions", params={"filters": "ACTIVE:1", "fields": "NAME,CITY,STALP,ASSET,DEP",
                                              "sort_by": "ASSET", "sort_order": "DESC", "limit": 5})
    fails = get(f"{BASE}/failures", params={"fields": "NAME,CITYST,FAILDATE,QBFASSET,COST", "sort_by": "FAILDATE",
                                            "sort_order": "DESC", "limit": 4})
    show({"active_insured_banks": big["meta"]["total"],
          "largest_by_assets_thousands_usd": [b["data"] for b in big["data"]],
          "recent_failures": [f["data"] for f in fails["data"]], "failures_since_1934": fails["meta"]["total"]})


if __name__ == "__main__":
    main()
