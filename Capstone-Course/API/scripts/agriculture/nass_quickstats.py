META = dict(
    name="USDA NASS Quick Stats",
    domain="Agriculture & Food",
    docs="https://quickstats.nass.usda.gov/api",
    auth="API key (query `key`)",
    env="NASS_API_KEY",
    signup="Request a free key at https://quickstats.nass.usda.gov/api (enter email; key arrives instantly). "
           "Max 50,000 records per call - use /api/get_counts/ first to size a query.",
    provides=["Official US agricultural statistics since the 1800s: crop acreage, yield, production, prices received",
              "Livestock inventories, dairy, poultry; Census of Agriculture (farms, land, operators, sales) every 5 years",
              "National, state, county and agricultural-district levels; weekly crop progress and condition"],
    ideas="Yield forecasting with weather/soil data, farm consolidation trends, climate impacts on county crop yields.",
    endpoint="GET https://quickstats.nass.usda.gov/api/api_GET/",
    params=[
        ("commodity_desc", "Commodity", "CORN"),
        ("statisticcat_desc", "AREA PLANTED | AREA HARVESTED | YIELD | PRODUCTION | PRICE RECEIVED | INVENTORY | ...", "YIELD"),
        ("agg_level_desc", "NATIONAL | STATE | COUNTY | AGRICULTURAL DISTRICT", "STATE"),
        ("state_alpha / county_name", "Location filters", "IA"),
        ("year / year__GE / year__LE", "Year, or a range with __GE / __LE suffixes", "year__GE=2015"),
        ("source_desc", "SURVEY | CENSUS", "SURVEY"),
        ("short_desc", "Exact data-item name (most precise filter)", "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"),
        ("format / key", "JSON | CSV | XML; your key", "JSON"),
        ("/api/get_counts/ · /api/get_param_values/?param=", "Sibling endpoints: record count for a query; valid values of a field", "param=commodity_desc"),
    ],
    sample="""[
  {"year": 2025, "state_name": "IOWA", "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value": "210.0"},
  {"year": 2024, "state_name": "IOWA", "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value": "211.0"},
  {"year": 2023, "state_name": "IOWA", "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value": "201.0"}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://quickstats.nass.usda.gov/api/api_GET/", params={
        "key": key("NASS_API_KEY"), "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
        "agg_level_desc": "STATE", "state_alpha": "IA", "source_desc": "SURVEY", "freq_desc": "ANNUAL",
        "reference_period_desc": "YEAR", "year__GE": 2021, "format": "JSON"})["data"]
    show([{k: r[k] for k in ("year", "state_name", "short_desc", "Value")} for r in sorted(d, key=lambda r: -r["year"])])


if __name__ == "__main__":
    main()
