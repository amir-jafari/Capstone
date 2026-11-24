META = dict(
    name="AviationWeather.gov Data API (METAR/TAF)",
    domain="Transportation & Geospatial",
    docs="https://aviationweather.gov/data/api/",
    auth="None",
    signup="No registration. Max 100 requests/minute; for bulk use the cache files (aviationweather.gov/data/cache).",
    provides=["METARs (hourly airport weather observations): temperature, wind, visibility, ceiling, flight category",
              "TAFs (airport forecasts), PIREPs (pilot reports), SIGMETs and AIRMETs (hazards)", "Station info for ~10,000 airports worldwide"],
    ideas="Weather-caused flight delay models (with BTS on-time data), fog/visibility forecasting, aviation-hazard maps.",
    endpoint="GET https://aviationweather.gov/api/data/{metar | taf | pirep | airsigmet | stationinfo}",
    params=[
        ("ids", "ICAO airport codes, comma list (or @state, e.g. @VA)", "KDCA,KIAD,KBWI"),
        ("format", "json | geojson | xml | raw", "json"),
        ("hours", "Hours back for METARs", "3"),
        ("bbox", "lat0,lon0,lat1,lon1 instead of ids", "38,-78,40,-76"),
        ("taf", "true -> include the TAF with METARs", "true"),
        ("fltcat (response)", "VFR | MVFR | IFR | LIFR flight category", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://aviationweather.gov/api/data/metar", params={"ids": "KDCA,KIAD,KBWI", "format": "json"})
    show([{"airport": m["name"], "time": m["reportTime"], "temp_C": m.get("temp"), "wind_kt": m.get("wspd"),
           "visibility": m.get("visib"), "flight_category": m.get("fltCat"), "raw": m["rawOb"]} for m in d])


if __name__ == "__main__":
    main()
