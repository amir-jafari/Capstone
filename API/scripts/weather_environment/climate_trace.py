META = dict(
    name="Climate TRACE (global emissions inventory)",
    domain="Weather & Climate",
    docs="https://api.climatetrace.org/v7/docs/index.html",
    auth="None",
    signup="No registration. Independent, satellite- and AI-based estimates of greenhouse-gas emissions for every "
           "country and hundreds of thousands of individual sources (power plants, oil fields, steel mills, ...).",
    provides=["Country rankings and totals by gas (CO2, CH4, N2O, CO2e) and sector, with per-capita figures and % change",
              "Individual emitting assets: location, sector, subsector, asset type, yearly emissions",
              "Sector/subsector definitions"],
    ideas="Verifying national climate pledges, top emitters by sector, mapping emissions sources against communities.",
    endpoint="GET https://api.climatetrace.org/v7/{resource}",
    params=[
        ("rankings/countries?gas=&start=&end=", "Country ranking for a period; add sectors= to restrict", "gas=co2e_100yr&start=2024&end=2024"),
        ("sources?gadmId=&sectors=&subsectors=&year=", "Individual sources; filter country with gadmId (country/countries are ignored)", "gadmId=USA&sectors=power"),
        ("limit / offset", "Paging", "5"),
        ("gas", "co2 | ch4 | n2o | co2e_100yr | co2e_20yr", "co2e_100yr"),
        ("definitions/sectors", "Sector and subsector names", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    r = get("https://api.climatetrace.org/v7/rankings/countries",
            params={"gas": "co2e_100yr", "start": 2024, "end": 2024, "limit": 5})
    plants = get("https://api.climatetrace.org/v7/sources",
                 params={"gadmId": "USA", "sectors": "power", "year": 2024, "limit": 3})
    show({"global_total_Gt_CO2e_2024": round(r["totals"]["emissionsQuantity"] / 1e9, 1), "top_countries": [
        {"country": c["name"], "Gt_CO2e": round(c["emissionsQuantity"] / 1e9, 2), "share_pct": round(c["percentage"], 1),
         "t_per_capita": round(c["emissionsPerCapita"], 1)} for c in r["rankings"]],
          "largest_US_power_sources": [{"name": s["name"], "type": s["assetType"],
                                        "Mt_CO2e": round(s["emissionsQuantity"] / 1e6, 1)} for s in plants]})


if __name__ == "__main__":
    main()
