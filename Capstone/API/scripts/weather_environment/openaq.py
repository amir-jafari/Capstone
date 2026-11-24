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
    endpoint="GET https://api.openaq.org/v3/locations",
    params=[
        ("coordinates", "lat,lon centre point", "38.9,-77.04"),
        ("radius", "Metres around coordinates (max 25,000)", "20000"),
        ("bbox", "minLon,minLat,maxLon,maxLat (alternative to coordinates)", "-77.2,38.8,-76.9,39.0"),
        ("iso", "Country ISO code", "US"),
        ("parameters_id", "Pollutant ID (e.g. 2 = PM2.5); list via /v3/parameters", "2"),
        ("limit / page", "Paging", "5"),
        ("/v3/sensors/{id}/measurements?datetime_from=&datetime_to=", "Sibling endpoint: time series for one sensor", "datetime_from=2026-10-01"),
        ("X-API-Key (header)", "Your key", "<OPENAQ_API_KEY>"),
    ],
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
