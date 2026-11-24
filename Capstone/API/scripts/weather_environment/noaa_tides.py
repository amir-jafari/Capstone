META = dict(
    name="NOAA Tides & Currents (CO-OPS)",
    domain="Weather & Climate",
    docs="https://api.tidesandcurrents.noaa.gov/api/prod/",
    auth="None",
    signup="No registration. Station IDs: https://tidesandcurrents.noaa.gov/map/ (8594900 = Washington, DC).",
    provides=["Observed water levels (6-minute), tide predictions, high/low tides at ~200 coastal stations",
              "Meteorological data at stations (air/water temperature, wind, pressure)", "Long-term sea level data (monthly means)"],
    ideas="Sea-level-rise trends, nuisance (high-tide) flooding frequency, storm surge vs. property values.",
    endpoint="GET https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
    params=[
        ("station", "Station ID", "8594900"),
        ("product", "water_level | predictions | hourly_height | high_low | monthly_mean | water_temperature | wind | ...", "monthly_mean"),
        ("date / begin_date & end_date", "latest | today | recent, or a YYYYMMDD range (limits vary by product)", "begin_date=20160101&end_date=20251231"),
        ("datum", "Vertical reference: MLLW | MSL | NAVD | STND ...", "MSL"),
        ("units / time_zone", "metric | english; gmt | lst | lst_ldt", "metric / gmt"),
        ("interval (predictions)", "h hourly, hilo highs/lows, or minutes", "hilo"),
        ("format", "json | csv | xml", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def main():
    common = {"station": "8594900", "units": "metric", "time_zone": "gmt", "format": "json"}
    latest = get(URL, params={**common, "product": "water_level", "datum": "MLLW", "date": "latest"})
    monthly = get(URL, params={**common, "product": "monthly_mean", "datum": "MSL",
                               "begin_date": "20150101", "end_date": "20251231"})["data"]
    by_year = {}
    for m in monthly:
        if m["MSL"]:
            by_year.setdefault(m["year"], []).append(float(m["MSL"]))
    show({"station": latest["metadata"]["name"], "latest_water_level_m_MLLW": latest["data"][0],
          "annual_mean_sea_level_m": {y: round(sum(v) / len(v), 3) for y, v in by_year.items() if len(v) == 12}},
         max_items=12)


if __name__ == "__main__":
    main()
