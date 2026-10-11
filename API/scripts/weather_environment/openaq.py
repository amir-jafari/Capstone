META = dict(
    name="OpenAQ v3 (Air Quality)",
    domain="Weather & Climate",
    docs="https://docs.openaq.org/",
    auth="API key (header `X-API-Key`)",
    env="OPENAQ_API_KEY",
    signup="Create a free account at https://explore.openaq.org/register -> Settings -> API Key. "
           "60 req/min, 2,000 req/hour.",
    provides=["Ground-station air-quality measurements worldwide: PM2.5, PM10, O3, NO2, SO2, CO",
              "Location metadata (provider, sensors, coordinates)", "Hourly / daily aggregates"],
    ideas="Environmental justice studies, pollution vs. health outcomes, wildfire smoke impact.",
    sample="""[
  {"location": "Washington DC - River Terrace", "country": "US", "sensors": ["pm25 µg/m³", "o3 ppm", "no2 ppm"]},
  {"location": "McMillan Reservoir", "country": "US", "sensors": ["pm25 µg/m³", "pm10 µg/m³"]}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.openaq.org/v3/locations", headers={"X-API-Key": key("OPENAQ_API_KEY")},
            params={"coordinates": "38.9,-77.04", "radius": 20000, "limit": 5})
    show([{"location": l["name"], "country": l["country"]["code"],
           "sensors": [f'{s["parameter"]["name"]} {s["parameter"]["units"]}' for s in l["sensors"]]}
          for l in d["results"]])


if __name__ == "__main__":
    main()
