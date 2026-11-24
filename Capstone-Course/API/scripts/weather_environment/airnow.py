META = dict(
    name="EPA AirNow API (real-time air quality)",
    domain="Weather & Climate",
    docs="https://docs.airnowapi.org/webservices",
    auth="API key (query `API_KEY`)",
    env="AIRNOW_API_KEY",
    signup="Request an account at https://docs.airnowapi.org/account/request/, then copy the key from the Web Services "
           "page. Real-time data are preliminary (use EPA AQS for verified history).",
    provides=["Current AQI by ZIP code or lat/lon for ozone, PM2.5 and PM10, with category (Good ... Hazardous)",
              "Daily AQI forecasts by reporting area", "Historical observations by date; monitoring-site data in a bounding box"],
    ideas="Wildfire-smoke exposure, AQI alerts vs. outdoor activity or hospital visits, forecast accuracy.",
    endpoint="GET https://www.airnowapi.org/aq/{observation | forecast}/{zipCode | latLong}/{current | historical}/",
    params=[
        ("zipCode / latitude & longitude", "Location", "20052"),
        ("distance", "Search radius in miles if no monitor at the location", "25"),
        ("date (forecast, historical)", "YYYY-MM-DD (historical: YYYY-MM-DDT00-0000)", "2026-10-12"),
        ("format", "application/json | text/csv | application/xml", "application/json"),
        ("API_KEY", "Your key", "<AIRNOW_API_KEY>"),
    ],
    sample="""[
  {"DateObserved": "2026-10-12", "HourObserved": 9, "ReportingArea": "Metropolitan Washington", "ParameterName": "O3", "AQI": 31, "Category": "Good"},
  {"DateObserved": "2026-10-12", "HourObserved": 9, "ReportingArea": "Metropolitan Washington", "ParameterName": "PM2.5", "AQI": 46, "Category": "Good"}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://www.airnowapi.org/aq/observation/zipCode/current/", params={
        "format": "application/json", "zipCode": "20052", "distance": 25, "API_KEY": key("AIRNOW_API_KEY")})
    show([{k: o[k] for k in ("DateObserved", "HourObserved", "ReportingArea", "ParameterName", "AQI")} |
          {"Category": o["Category"]["Name"]} for o in d])


if __name__ == "__main__":
    main()
