META = dict(
    name="Our World in Data — Energy dataset",
    domain="Engineering, Energy & Tech",
    docs="https://github.com/owid/energy-data",
    auth="None (CSV on GitHub)",
    signup="No registration. One tidy country-year CSV (~10 MB) combining the Energy Institute Statistical Review, "
           "Ember and EIA data; codebook in the repo. CC-BY.",
    provides=["Primary energy and electricity by source (coal, gas, oil, nuclear, hydro, wind, solar, biofuels) for every country",
              "Shares of renewables/low-carbon, per-capita consumption, carbon intensity of electricity, energy per GDP",
              "Years 1900-present (coverage varies by variable)"],
    ideas="Energy-transition tracking, renewable adoption drivers, decoupling of energy use from growth.",
    endpoint="GET https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
    params=[
        ("key columns", "country, year, iso_code, population, gdp", "—"),
        ("{source}_share_elec", "Share of electricity from a source (%)", "solar_share_elec, wind_share_elec"),
        ("{source}_consumption / {source}_electricity", "Primary energy (TWh) / electricity generation (TWh)", "coal_electricity"),
        ("carbon_intensity_elec", "gCO2 per kWh of electricity", "—"),
        ("owid-energy-codebook.csv", "Variable definitions and sources (same repo)", "—"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    rows = csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
                                          as_json=False)))
    keep = {"United States", "China", "Germany", "India", "World"}
    latest = {}
    for r in rows:  # latest year with a value for each country
        if r["country"] in keep and r["renewables_share_elec"]:
            latest[r["country"]] = r
    show([{"country": c, "year": r["year"], "renewables_share_elec_pct": round(float(r["renewables_share_elec"]), 1),
           "solar_pct": round(float(r["solar_share_elec"] or 0), 1), "wind_pct": round(float(r["wind_share_elec"] or 0), 1),
           "coal_pct": round(float(r["coal_share_elec"] or 0), 1),
           "carbon_intensity_g_per_kWh": r["carbon_intensity_elec"] and round(float(r["carbon_intensity_elec"]))}
          for c, r in sorted(latest.items())])


if __name__ == "__main__":
    main()
