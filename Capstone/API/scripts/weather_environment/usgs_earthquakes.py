META = dict(
    name="USGS Earthquake Catalog",
    domain="Weather & Climate",
    docs="https://earthquake.usgs.gov/fdsnws/event/1/",
    auth="None",
    signup="No registration.",
    provides=["Global earthquakes with magnitude, depth, location, time, tsunami flag, felt reports",
              "GeoJSON output; real-time feeds and full historical catalog"],
    ideas="Spatio-temporal clustering, aftershock forecasting, geo-visualization.",
    endpoint="GET https://earthquake.usgs.gov/fdsnws/event/1/query",
    params=[
        ("format", "geojson | csv | xml | text | kml", "geojson"),
        ("starttime / endtime", "ISO-8601 dates (default: last 30 days)", "2026-01-01"),
        ("minmagnitude / maxmagnitude", "Magnitude range", "5"),
        ("latitude / longitude / maxradiuskm", "Circle search", "38.9 / -77.0 / 500"),
        ("minlatitude / maxlatitude / minlongitude / maxlongitude", "Bounding-box search", "24 / 50 / -125 / -66"),
        ("orderby", "time | time-asc | magnitude | magnitude-asc", "time"),
        ("limit / offset", "Max 20,000 events per request", "5"),
    ],
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
