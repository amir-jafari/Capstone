META = dict(
    name="U.S. Bureau of Economic Analysis (BEA) API",
    domain="Finance & Economics",
    docs="https://apps.bea.gov/api/signup/",
    auth="API key (query `UserID`)",
    env="BEA_API_KEY",
    signup="Register at https://apps.bea.gov/API/signup/ - the 36-character key is emailed and must be activated from "
           "the email link. Limits: 100 requests and 100 MB per minute. An invalid key still returns HTTP 200 with an "
           "error inside BEAAPI.Results.Error, so check for it.",
    provides=["National accounts (NIPA): GDP and components, personal income, PCE prices, corporate profits",
              "Regional: GDP, personal income and employment by state, county and metro area",
              "Industry GDP, international transactions, trade in services, direct investment"],
    ideas="State/metro growth comparisons, regional income inequality, industry contribution to GDP.",
    endpoint="GET https://apps.bea.gov/api/data",
    params=[
        ("method", "GetData | GetDataSetList | GetParameterList | GetParameterValues", "GetData"),
        ("DataSetName", "NIPA | Regional | GDPbyIndustry | ITA | IntlServTrade | ...", "Regional"),
        ("TableName / LineCode", "Table and line (Regional: SAGDP1 state GDP, CAINC1 county income; NIPA: T10101 real GDP growth)", "SAGDP1 / 1"),
        ("GeoFips", "STATE | COUNTY | MSA | a FIPS code", "STATE"),
        ("Year / Frequency", "Year list or ALL / LAST5; A | Q | M", "2024 / A"),
        ("ResultFormat / UserID", "JSON | XML; your key", "JSON"),
    ],
    sample="""[
  {"state": "California", "year": "2024", "gdp_millions_usd": 4103124.4},
  {"state": "Texas", "year": "2024", "gdp_millions_usd": 2709393.0},
  {"state": "New York", "year": "2024", "gdp_millions_usd": 2297028.1}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://apps.bea.gov/api/data", params={
        "UserID": key("BEA_API_KEY"), "method": "GetData", "DataSetName": "Regional", "TableName": "SAGDP1",
        "LineCode": 3, "GeoFips": "STATE", "Year": 2024, "ResultFormat": "JSON"})["BEAAPI"]["Results"]
    if "Error" in d:  # BEA reports errors with HTTP 200
        raise SystemExit(d["Error"])
    rows = sorted(d["Data"], key=lambda r: -float(r["DataValue"].replace(",", "")))[:5]
    show([{"state": r["GeoName"], "year": r["TimePeriod"], "gdp_millions_usd": float(r["DataValue"].replace(",", ""))}
          for r in rows])


if __name__ == "__main__":
    main()
