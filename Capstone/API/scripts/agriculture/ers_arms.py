META = dict(
    name="USDA ERS — ARMS Farm Financial Data",
    domain="Agriculture & Food",
    docs="https://www.ers.usda.gov/developer/data-apis/arms-data-api",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use your api.data.gov key (https://api.data.gov/signup/). The Agricultural Resource Management Survey (ARMS) "
           "is USDA's main source on farm finances and household income.",
    provides=["Farm income statements and balance sheets (revenue, expenses, net farm income, assets, debt)",
              "Breakdowns by farm typology, size, commodity specialty, region and state", "Farm household income and operator characteristics"],
    ideas="Farm financial stress, small vs. large farm economics, policy (subsidy) incidence across farm types.",
    endpoint="GET https://api.ers.usda.gov/data/arms/{surveydata | year | state | report | variable | farmtype | category}",
    params=[
        ("year", "Survey year(s)", "2023"),
        ("report", "Report name: income statement, balance sheet, farm business debt, operator household income, ...", "income statement"),
        ("variable", "Variable ID(s) (list via /data/arms/variable)", "kount"),
        ("farmtype / category / category_value", "Farm type and breakdown category (list via /farmtype, /category)", "category=Farm Typology"),
        ("state", "State name or 'all'", "all"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.ers.usda.gov/data/arms/surveydata", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "year": 2023, "report": "income statement",
        "variable": "kount", "category": "Farm Typology"})
    show({"report": "Farm Business Income Statement (2023), estimated number of farms", "by_farm_type": [
        {"farm_type": r["categoryValue"], "farms": r["estimate"], "rse_pct": r["rse"]} for r in d["data"]]})


if __name__ == "__main__":
    main()
