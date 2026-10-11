META = dict(
    name="USGS Earthquake Catalog",
    domain="Weather & Climate",
    docs="https://earthquake.usgs.gov/fdsnws/event/1/",
    auth="None",
    signup="No registration.",
    provides=["Global earthquakes with magnitude, depth, location, time, tsunami flag, felt reports",
              "GeoJSON output; real-time feeds and full historical catalog"],
    ideas="Spatio-temporal clustering, aftershock forecasting, geo-visualization.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
        "format": "geojson", "minmagnitude": 5, "orderby": "time", "limit": 5})
    show([{"mag": f["properties"]["mag"], "place": f["properties"]["place"],
           "time_ms": f["properties"]["time"], "depth_km": f["geometry"]["coordinates"][2],
           "tsunami": f["properties"]["tsunami"]} for f in d["features"]])


if __name__ == "__main__":
    main()
