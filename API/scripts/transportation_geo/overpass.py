META = dict(
    name="OpenStreetMap Overpass API",
    domain="Transportation & Geospatial",
    docs="https://wiki.openstreetmap.org/wiki/Overpass_API",
    auth="None",
    signup="No registration. Keep queries small; heavy users should self-host.",
    provides=["Query any OSM feature in an area: roads, bike lanes, bus stops, restaurants, schools, hospitals, buildings",
              "Tags (name, cuisine, opening_hours, amenity...) and coordinates/geometry"],
    ideas="Food deserts, walkability scores, amenity access, urban-form features for ML.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show
import requests


def main():
    q = """[out:json][timeout:25];
    node["amenity"="cafe"](around:600,38.8997,-77.0486);
    out 5;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                      headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    show([{"name": e["tags"].get("name"), "lat": e["lat"], "lon": e["lon"],
           "opening_hours": e["tags"].get("opening_hours")} for e in r.json()["elements"]])


if __name__ == "__main__":
    main()
