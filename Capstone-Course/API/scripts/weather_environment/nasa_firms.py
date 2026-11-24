META = dict(
    name="NASA FIRMS (active fire detections)",
    domain="Weather & Climate",
    docs="https://firms.modaps.eosdis.nasa.gov/api/",
    auth="MAP_KEY in the URL path",
    env="FIRMS_MAP_KEY",
    signup="Get a free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/ (email). "
           "Limit: 5,000 transactions per 10 minutes.",
    provides=["Satellite fire/thermal-anomaly detections (VIIRS 375 m, MODIS 1 km) within ~3 hours of overpass",
              "Latitude/longitude, brightness, fire radiative power (FRP), confidence, day/night, acquisition time",
              "Archive back to 2000 (MODIS) / 2012 (VIIRS)"],
    ideas="Wildfire spread analysis, smoke-exposure modelling (with AirNow/OpenAQ), agricultural burning detection.",
    endpoint="GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{area}/{day_range}[/{date}]",
    params=[
        ("source", "VIIRS_SNPP_NRT | VIIRS_NOAA20_NRT | VIIRS_NOAA21_NRT | MODIS_NRT (and _SP for standard/archive)", "VIIRS_SNPP_NRT"),
        ("area", "world, or a bounding box west,south,east,north", "-125,24,-66,50"),
        ("day_range", "1-10 days", "1"),
        ("date", "Optional start date YYYY-MM-DD (default: most recent)", "2026-08-15"),
        ("/api/country/csv/{MAP_KEY}/{source}/{ISO3}/{days}", "Sibling endpoint: detections in one country", "/USA/1"),
    ],
    sample="""{
  "detections_last_24h_contiguous_US": 1843,
  "highest_FRP": [
    {"latitude": 41.21, "longitude": -122.53, "acq_date": "2026-10-11", "acq_time": "2042", "frp": 412.7, "confidence": "h", "daynight": "D"}
  ]
}""",
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    text = get(f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key("FIRMS_MAP_KEY")}/VIIRS_SNPP_NRT/-125,24,-66,50/1',
               as_json=False)
    rows = list(csv.DictReader(io.StringIO(text)))
    top = sorted(rows, key=lambda r: -float(r["frp"]))[:5]
    show({"detections_last_24h_contiguous_US": len(rows), "highest_FRP": [
        {k: r[k] for k in ("latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight")} for r in top]})


if __name__ == "__main__":
    main()
