META = dict(
    name="NASA Open APIs (APOD, NeoWs, DONKI, EONET...)",
    domain="Science & Research",
    docs="https://api.nasa.gov/",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Get a key instantly at https://api.nasa.gov/ (same api.data.gov key). 1,000 req/hour; DEMO_KEY 30/hour.",
    provides=["Near-Earth asteroids: size, velocity, miss distance, hazard flag (NeoWs)",
              "Astronomy Picture of the Day, Mars rover photos, space weather (DONKI)",
              "Natural events: wildfires, storms, volcanoes (EONET, keyless)"],
    ideas="Asteroid hazard classification, wildfire event tracking, image datasets.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    d = get("https://api.nasa.gov/neo/rest/v1/feed", params={
        "start_date": "2026-10-01", "end_date": "2026-10-01", "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")})
    neos = d["near_earth_objects"]["2026-10-01"]
    show({"count": d["element_count"], "asteroids": [
        {"name": n["name"], "diameter_m_max": round(n["estimated_diameter"]["meters"]["estimated_diameter_max"]),
         "hazardous": n["is_potentially_hazardous_asteroid"],
         "miss_km": round(float(n["close_approach_data"][0]["miss_distance"]["kilometers"])),
         "velocity_kph": round(float(n["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]))}
        for n in neos]})


if __name__ == "__main__":
    main()
