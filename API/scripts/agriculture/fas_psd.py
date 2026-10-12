META = dict(
    name="USDA FAS — Production, Supply & Distribution (PSD) and trade",
    domain="Agriculture & Food",
    docs="https://apps.fas.usda.gov/opendatawebv2/#/home",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use your api.data.gov key (https://api.data.gov/signup/). The old apps.fas.usda.gov/OpenData endpoints now "
           "return 403 - use api.fas.usda.gov as below.",
    provides=["PSD: official USDA estimates of production, consumption, trade and stocks for ~60 commodities in every country",
              "Export Sales Reporting (ESR): weekly US export sales by commodity and destination",
              "GATS: US agricultural trade from Census (exports/imports by partner and product)"],
    ideas="Global grain-market balance and price forecasting, food-security exposure to export shocks, trade-war effects.",
    endpoint="GET https://api.fas.usda.gov/api/psd/commodity/{commodityCode}/{world | country/{code} | country/all}/year/{marketYear}",
    params=[
        ("commodityCode", "From /api/psd/commodities (e.g. 0440000 corn, 0410000 wheat, 2222000 soybeans)", "0440000"),
        ("world / country/{code} / country/all", "World total, one country, or every country", "world"),
        ("marketYear", "Marketing year", "2024"),
        ("/api/psd/commodityAttributes · /unitsOfMeasure · /countries", "Lookups for attributeId, unitId, countryCode", "—"),
        ("/api/esr/exports/commodityCode/{c}/allCountries/marketYear/{y}", "Weekly export sales", "—"),
        ("/api/gats/censusExports/partnerCode/{p}/year/{y}/month/{m}", "Monthly US exports to a partner", "partnerCode/CH/year/2024/month/1"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://api.fas.usda.gov/api/psd"


def main():
    k = {"api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")}
    attrs = {a["attributeId"]: a["attributeName"].strip() for a in get(f"{BASE}/commodityAttributes", params=k)}
    units = {u["unitId"]: u["unitDescription"].strip() for u in get(f"{BASE}/unitsOfMeasure", params=k)}
    rows = get(f"{BASE}/commodity/0440000/world/year/2024", params=k)  # corn, world total
    show({"commodity": "Corn, world, market year 2024", "balance": [
        {"attribute": attrs.get(r["attributeId"], r["attributeId"]), "value": r["value"], "unit": units.get(r["unitId"])}
        for r in rows]}, max_items=12)


if __name__ == "__main__":
    main()
