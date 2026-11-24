META = dict(
    name="FHFA House Price Index",
    domain="Housing & Real Estate",
    docs="https://www.fhfa.gov/data/hpi",
    auth="None (CSV download)",
    signup="No registration; no REST API. The master CSV (~17 MB) holds every published index; smaller files by "
           "geography are on the docs page. Many series are also on FRED (needs a FRED key).",
    provides=["Repeat-sales house price indexes since 1975 (purchase-only, all-transactions, expanded-data)",
              "National, census division, state, metro (MSA), and (annually) county, ZIP and tract levels",
              "Seasonally adjusted and not adjusted"],
    ideas="Long-run price appreciation by region, house prices vs. local employment, housing-wealth effects.",
    endpoint="GET https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv",
    params=[
        ("hpi_type", "traditional | distress-free | ...", "traditional"),
        ("hpi_flavor", "purchase-only | all-transactions | expanded-data", "purchase-only"),
        ("frequency", "monthly | quarterly", "monthly"),
        ("level", "USA or Census Division | State | MSA | ...", "USA or Census Division"),
        ("place_name / place_id", "Geography name / code", "United States / USA"),
        ("yr / period / index_nsa / index_sa", "Year, month or quarter, index values (1991 = 100 for monthly)", "2026 / 7"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    rows = csv.DictReader(io.StringIO(get("https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv", as_json=False)))
    us = [r for r in rows if r["place_id"] == "USA" and r["hpi_flavor"] == "purchase-only" and r["frequency"] == "monthly"]
    latest, year_ago = us[-1], us[-13]
    show({"series": "US purchase-only HPI, seasonally adjusted (Jan 1991 = 100)",
          "latest": [{"month": f'{r["yr"]}-{int(r["period"]):02d}', "index_sa": float(r["index_sa"])} for r in us[-4:]],
          "yoy_change_pct": round(100 * (float(latest["index_sa"]) / float(year_ago["index_sa"]) - 1), 2)})


if __name__ == "__main__":
    main()
