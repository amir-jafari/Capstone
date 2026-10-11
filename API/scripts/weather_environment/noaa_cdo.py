META = dict(
    name="NOAA Climate Data Online (CDO)",
    domain="Weather & Climate",
    docs="https://www.ncdc.noaa.gov/cdo-web/webservices/v2",
    auth="Token (header `token`)",
    env="NOAA_TOKEN",
    signup="Request a token at https://www.ncdc.noaa.gov/cdo-web/token (enter email; token is emailed). "
           "5 req/sec, 10,000 req/day.",
    provides=["Daily/monthly station observations (GHCND): TMAX, TMIN, PRCP, SNOW",
              "Station metadata, datasets back to the 1800s"],
    ideas="Long-term local climate trends, heat-wave studies, weather/health linkage.",
    sample="""[
  {"date": "2025-07-01T00:00:00", "datatype": "TMAX", "station": "GHCND:USW00013743", "value": 33.9},
  {"date": "2025-07-01T00:00:00", "datatype": "TMIN", "station": "GHCND:USW00013743", "value": 24.4},
  {"date": "2025-07-02T00:00:00", "datatype": "TMAX", "station": "GHCND:USW00013743", "value": 35.0}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://www.ncei.noaa.gov/cdo-web/api/v2/data", headers={"token": key("NOAA_TOKEN")}, params={
        "datasetid": "GHCND", "stationid": "GHCND:USW00013743",  # Reagan National Airport
        "datatypeid": "TMAX,TMIN", "startdate": "2025-07-01", "enddate": "2025-07-07", "units": "metric",
    })
    show([{k: r[k] for k in ("date", "datatype", "station", "value")} for r in d["results"]])


if __name__ == "__main__":
    main()
