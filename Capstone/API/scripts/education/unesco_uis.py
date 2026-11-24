META = dict(
    name="UNESCO Institute for Statistics (UIS) Data API",
    domain="Education",
    docs="https://api.uis.unesco.org/api/public/documentation/",
    auth="None",
    signup="No registration (fair-use limits). The official global source for SDG 4 education indicators.",
    provides=["Enrollment, completion, out-of-school rates, literacy, learning outcomes, teachers, education spending",
              "Every country and region, ~1970-today, plus science/culture indicators"],
    ideas="Global education gaps, spending vs. outcomes, gender parity in schooling.",
    endpoint="GET https://api.uis.unesco.org/api/public/data/indicators",
    params=[
        ("indicator", "Indicator ID(s); browse via /api/public/definitions/indicators", "CR.1 (completion rate, primary)"),
        ("geoUnit", "ISO3 country or region code(s)", "USA"),
        ("geoUnitType", "NATIONAL | REGIONAL (instead of geoUnit)", "NATIONAL"),
        ("start / end", "Year range", "2015 / 2024"),
        ("indicatorMetadata / footnotes", "true → include definitions and footnotes", "true"),
        ("/api/public/versions", "Data release versions (pin a version for reproducibility)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.uis.unesco.org/api/public/data/indicators", params={
        "indicator": "CR.1", "geoUnit": ["USA", "IND", "NGA", "BRA", "DEU"], "start": 2015})
    latest = {}
    for r in d["records"]:  # keep the most recent year per country
        if r["value"] is not None and r["year"] >= latest.get(r["geoUnit"], {}).get("year", 0):
            latest[r["geoUnit"]] = r
    show([{"country": g, "year": r["year"], "primary_completion_rate_pct": round(r["value"], 1)}
          for g, r in latest.items()])


if __name__ == "__main__":
    main()
