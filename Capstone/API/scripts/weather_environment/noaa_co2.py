META = dict(
    name="NOAA Global Monitoring Lab — Mauna Loa CO2",
    domain="Weather & Climate",
    docs="https://gml.noaa.gov/ccgg/trends/data.html",
    auth="None (CSV/text files)",
    signup="No registration; no REST API - stable file URLs updated monthly. Also global-average CO2, methane (CH4), "
           "N2O and SF6 files on the same page.",
    provides=["Monthly mean atmospheric CO2 at Mauna Loa since March 1958 (the Keeling Curve), plus deseasonalized trend",
              "Daily, weekly and annual files; global marine-surface averages; annual growth rates"],
    ideas="Time-series decomposition and forecasting, teaching seasonality and trend.",
    endpoint="GET https://gml.noaa.gov/webdata/ccgg/trends/{gas}/{file}.csv",
    params=[
        ("co2/co2_mm_mlo.csv", "Monthly Mauna Loa: year, month, decimal date, average, deseasonalized, ...", "co2_mm_mlo.csv"),
        ("co2/co2_annmean_mlo.csv", "Annual means", "co2_annmean_mlo.csv"),
        ("co2/co2_mm_gl.csv", "Global monthly mean", "co2_mm_gl.csv"),
        ("ch4/ch4_mm_gl.csv / n2o/n2o_mm_gl.csv", "Global methane / nitrous oxide", "ch4_mm_gl.csv"),
        ("comment lines", "Header lines start with # - skip them", "—"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    text = get("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", as_json=False)
    rows = list(csv.DictReader(io.StringIO("\n".join(l for l in text.splitlines() if not l.startswith("#")))))
    latest, ten_years = rows[-1], rows[-121]
    show({"latest": [{"month": f'{r["year"]}-{int(r["month"]):02d}', "co2_ppm": float(r["average"]),
                      "deseasonalized_ppm": float(r["deseasonalized"])} for r in rows[-3:]],
          "first_record": f'{rows[0]["year"]}-{int(rows[0]["month"]):02d}: {rows[0]["average"]} ppm',
          "change_over_10_years_ppm": round(float(latest["deseasonalized"]) - float(ten_years["deseasonalized"]), 1)})


if __name__ == "__main__":
    main()
