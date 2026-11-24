META = dict(
    name="EPA Air Quality System (AQS) API",
    domain="Weather & Climate",
    docs="https://aqs.epa.gov/aqsweb/documents/data_api.html",
    auth="Email + key (query `email`, `key`)",
    env="AQS_EMAIL, AQS_KEY",
    signup="Sign up by calling https://aqs.epa.gov/data/api/signup?email=you@gwu.edu - the key is emailed to you. "
           "Limits: 10 requests/minute and 1 year of data per request; pause 5 seconds between calls.",
    provides=["Verified historical monitor data (the regulatory record) for criteria pollutants and many air toxics",
              "Sample, daily and annual summaries by site, county, state, CBSA or bounding box",
              "Monitor metadata and quality-assurance data"],
    ideas="Long-run pollution trends, health-outcome studies needing verified data, Clean Air Act policy evaluation.",
    endpoint="GET https://aqs.epa.gov/data/api/{service}/{geography}",
    params=[
        ("service", "sampleData | dailyData | annualData | monitors | list", "dailyData"),
        ("geography", "bySite | byCounty | byState | byBox | byCBSA", "byState"),
        ("param", "Parameter code(s): 88101 PM2.5, 44201 ozone, 42602 NO2 (up to 5)", "88101"),
        ("bdate / edate", "YYYYMMDD, within the same year", "20250101 / 20251231"),
        ("state / county / site", "FIPS codes", "state=11"),
        ("email / key", "Your credentials", "—"),
        ("list/states, list/parametersByClass", "Lookup lists for codes", "list/parametersByClass?pc=CRITERIA"),
    ],
    sample="""[
  {"date_local": "2025-07-04", "site_number": "0043", "local_site_name": "McMillan Reservoir", "parameter": "PM2.5 - Local Conditions", "arithmetic_mean": 18.4, "aqi": 64},
  {"date_local": "2025-07-05", "site_number": "0043", "local_site_name": "McMillan Reservoir", "parameter": "PM2.5 - Local Conditions", "arithmetic_mean": 9.1, "aqi": 50}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://aqs.epa.gov/data/api/dailyData/byState", params={
        "email": key("AQS_EMAIL"), "key": key("AQS_KEY"), "param": "88101",
        "bdate": "20250704", "edate": "20250705", "state": "11"})
    show([{k: r.get(k) for k in ("date_local", "site_number", "local_site_name", "parameter", "arithmetic_mean", "aqi")}
          for r in d["Data"]])


if __name__ == "__main__":
    main()
