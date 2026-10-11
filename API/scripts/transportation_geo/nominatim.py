META = dict(
    name="OpenStreetMap Nominatim (geocoding)",
    domain="Transportation & Geospatial",
    docs="https://nominatim.org/release-docs/latest/api/Overview/",
    auth="None (User-Agent required)",
    signup="No registration. Max 1 req/sec; no bulk geocoding on the public server (self-host for bulk).",
    provides=["Address -> lat/lon (forward geocoding) and lat/lon -> address (reverse)",
              "OSM class/type, bounding box, structured address parts"],
    ideas="Geocoding addresses in any dataset before spatial joins with Census tracts.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://nominatim.openstreetmap.org/search",
            params={"q": "George Washington University, Washington DC", "format": "jsonv2", "limit": 2})
    show([{k: r[k] for k in ("display_name", "lat", "lon", "category", "type")} for r in d])


if __name__ == "__main__":
    main()
