META = dict(
    name="Open-Meteo",
    domain="Weather & Climate",
    docs="https://open-meteo.com/en/docs",
    auth="None",
    signup="No registration. Free for non-commercial use (<10,000 calls/day).",
    provides=["Hourly/daily forecasts up to 16 days for any lat/lon",
              "Historical weather back to 1940 (ERA5 reanalysis) via archive-api.open-meteo.com",
              "Air quality, marine, flood, climate-change (CMIP6) projections, geocoding"],
    ideas="Weather as a covariate for sales/energy/mobility models; climate trend analysis.",
    endpoint="GET https://api.open-meteo.com/v1/forecast  ·  GET https://archive-api.open-meteo.com/v1/archive",
    params=[
        ("latitude / longitude", "Location (comma lists allowed for multiple points)", "38.90 / -77.04"),
        ("current", "Current-condition variables", "temperature_2m,wind_speed_10m"),
        ("hourly", "Hourly variables", "temperature_2m,precipitation"),
        ("daily", "Daily aggregates (requires timezone)", "temperature_2m_max,precipitation_sum"),
        ("timezone", "IANA zone or auto", "America/New_York"),
        ("forecast_days / past_days", "Forecast length 1-16 days / include recent past days", "3 / 7"),
        ("temperature_unit / wind_speed_unit / precipitation_unit", "celsius|fahrenheit, kmh|ms|mph|kn, mm|inch", "fahrenheit"),
        ("start_date / end_date", "Archive endpoint: historical range back to 1940", "1990-01-01 / 1990-12-31"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": 38.90, "longitude": -77.04,  # Washington, DC
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/New_York", "forecast_days": 3,
    })
    show({"current": d["current"], "daily": d["daily"]})


if __name__ == "__main__":
    main()
