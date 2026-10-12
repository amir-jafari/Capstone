META = dict(
    name="NOAA Climate at a Glance",
    domain="Weather & Climate",
    docs="https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/",
    auth="None",
    signup="No registration. Build a query on the website, then swap the page URL's ending for /data.json or /data.csv.",
    provides=["Temperature, precipitation, drought indices (PDSI) and degree days since 1895",
              "National (contiguous US), regional, state, climate-division, county and city series; global anomalies",
              "Any time scale: 1-month to 60-month periods"],
    ideas="Local warming trends for any US county, linking climate anomalies to crop yields, energy demand or health.",
    endpoint="GET https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/{scope}/time-series/{location}/{parameter}/{timescale}/{month}/{start}-{end}/data.json",
    params=[
        ("scope / location", "national/110 (contiguous US), statewide/{state code}, county/{state-county}, global/globe/land_ocean", "national/110"),
        ("parameter", "tavg | tmax | tmin | pcp (precipitation) | pdsi | cdd | hdd", "tavg"),
        ("timescale", "Months per period: 1, 3, 12, ...", "12"),
        ("month", "Ending month (0 = all months)", "0"),
        ("start-end", "Year range", "1895-2025"),
        ("data.json / data.csv", "Output format", "data.json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/12/12/1895-2025/data.json")
    years = {k[:4]: v["value"] for k, v in d["data"].items()}  # 12-month periods ending in December = calendar years
    warmest = sorted(years.items(), key=lambda kv: -kv[1])[:5]
    show({"series": d["description"]["title"] + f' ({d["description"]["units"]})',
          "warmest_years": [{"year": y, "avg_temp_F": t} for y, t in warmest],
          "first_decade_avg": round(sum(list(years.values())[:10]) / 10, 2),
          "last_decade_avg": round(sum(list(years.values())[-10:]) / 10, 2)})


if __name__ == "__main__":
    main()
