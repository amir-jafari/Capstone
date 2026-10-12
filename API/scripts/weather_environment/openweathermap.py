META = dict(
    name="OpenWeatherMap",
    domain="Weather & Climate",
    docs="https://openweathermap.org/api",
    auth="API key (query `appid`)",
    env="OPENWEATHER_API_KEY",
    signup="Sign up at https://home.openweathermap.org/users/sign_up -> 'API keys' tab. Free plan: "
           "60 calls/min, 1M calls/month (current weather, 5-day forecast, air pollution). "
           "New keys take ~10-60 min to activate.",
    provides=["Current weather by city / lat-lon / zip", "5-day / 3-hour forecast",
              "Air pollution (AQI, PM2.5, O3...)", "Geocoding"],
    ideas="Quick weather features for any global city; air-quality dashboards.",
    sample="""{
  "city": "Washington",
  "temp_C": 18.4,
  "humidity": 62,
  "description": "scattered clouds",
  "wind_mps": 3.6
}""",
    endpoint="GET https://api.openweathermap.org/data/2.5/weather",
    params=[
        ("q", "City name, state code, country code", "Washington,DC,US"),
        ("lat / lon", "Coordinates (alternative to q)", "38.9 / -77.04"),
        ("zip", "ZIP code, country code", "20052,US"),
        ("units", "standard (Kelvin) | metric | imperial", "metric"),
        ("lang", "Description language", "en"),
        ("appid", "Your API key", "<OPENWEATHER_API_KEY>"),
        ("/data/2.5/forecast", "Sibling endpoint: 5-day/3-hour forecast (same params, cnt= to limit)", "cnt=8"),
        ("/data/2.5/air_pollution?lat=&lon=", "Sibling endpoint: AQI and pollutant concentrations", "lat=38.9&lon=-77.04"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.openweathermap.org/data/2.5/weather",
            params={"q": "Washington,DC,US", "units": "metric", "appid": key("OPENWEATHER_API_KEY")})
    show({"city": d["name"], "temp_C": d["main"]["temp"], "humidity": d["main"]["humidity"],
          "description": d["weather"][0]["description"], "wind_mps": d["wind"]["speed"]})


if __name__ == "__main__":
    main()
