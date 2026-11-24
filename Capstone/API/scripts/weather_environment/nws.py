META = dict(
    name="US National Weather Service (api.weather.gov)",
    domain="Weather & Climate",
    docs="https://www.weather.gov/documentation/services-web-api",
    auth="None (User-Agent with contact required)",
    signup="No registration. Just send a User-Agent such as `(myapp, me@gwu.edu)`.",
    provides=["Official NWS forecasts (12-hour periods & hourly) for US locations",
              "Active weather alerts (watches/warnings) by state or point", "Station observations"],
    ideas="Severe-weather alert analysis, linking alerts to outages/traffic/311 calls.",
    endpoint="GET https://api.weather.gov/points/{lat},{lon}  ·  GET https://api.weather.gov/alerts/active",
    params=[
        ("points/{lat},{lon}", "Resolves a location to its forecast office/grid; returns forecast URLs", "38.8977,-77.0365"),
        ("forecast / forecastHourly", "URLs returned in points → properties (12-hour periods / hourly)", "properties.forecast"),
        ("alerts/active?area=", "State/marine area code", "VA"),
        ("alerts/active?severity=", "Extreme | Severe | Moderate | Minor", "Severe"),
        ("alerts/active?event=", "Alert event name", "Flood Warning"),
        ("stations/{id}/observations/latest", "Latest observation at a station", "KDCA"),
        ("User-Agent (header)", "Required: app name + contact", "(myapp, me@gwu.edu)"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    point = get("https://api.weather.gov/points/38.8977,-77.0365")["properties"]
    periods = get(point["forecast"])["properties"]["periods"][:3]
    alerts = get("https://api.weather.gov/alerts/active", params={"area": "VA"})["features"]
    show({
        "office": point["gridId"],
        "forecast": [{k: p[k] for k in ("name", "temperature", "temperatureUnit", "shortForecast")} for p in periods],
        "active_alerts_VA": [a["properties"]["headline"] for a in alerts[:3]],
    })


if __name__ == "__main__":
    main()
