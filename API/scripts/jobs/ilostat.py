META = dict(
    name="ILOSTAT (International Labour Organization)",
    domain="Jobs & Labor",
    docs="https://ilostat.ilo.org/data/api/",
    auth="None",
    signup="No registration. The ILO's global labor statistics: harmonized estimates for ~190 countries. Two APIs: an "
           "SDMX endpoint (sdmx.ilo.org) and a simpler R/plumber JSON endpoint (used here).",
    provides=["Unemployment, employment, labor-force participation by sex, age, education",
              "Wages, working hours, informality, child labour, occupational injuries, labor income share",
              "Modelled estimates that fill gaps across countries and years"],
    ideas="Cross-country labor-market comparisons, youth unemployment drivers, informality vs. development.",
    endpoint="GET https://rplumber.ilo.org/data/indicator/",
    params=[
        ("id", "Indicator + frequency, e.g. UNE_DEAP_SEX_AGE_RT_A (unemployment rate, annual)", "UNE_DEAP_SEX_AGE_RT_A"),
        ("ref_area", "ISO3 country code(s), + separated", "USA+DEU+ZAF+IND"),
        ("sex / classif1", "SEX_T | SEX_M | SEX_F; age/education breakdowns e.g. AGE_YTHADULT_Y15-24", "SEX_T"),
        ("timefrom / timeto", "Year range", "2015 / 2025"),
        ("format", ".json | .csv", ".json"),
        ("sdmx.ilo.org/rest/data/ILO,{dataflow}/{key}", "SDMX alternative", "ILO,DF_UNE_DEAP_SEX_AGE_RT/USA.A..SEX_T.AGE_YTHADULT_YGE15"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://rplumber.ilo.org/data/indicator/", params={
        "id": "UNE_DEAP_SEX_AGE_RT_A", "ref_area": "USA+DEU+ZAF+IND+BRA", "sex": "SEX_T", "timefrom": 2024,
        "format": ".json"})
    rows = [r for r in d if r["classif1"] in ("AGE_YTHADULT_YGE15", "AGE_YTHADULT_Y15-24")]
    out = {}
    for r in sorted(rows, key=lambda r: r["time"]):
        label = "youth_15_24" if r["classif1"].endswith("Y15-24") else "all_15plus"
        out.setdefault(r["ref_area"], {})[label] = f'{round(r["obs_value"], 1)}% ({r["time"]})'
    show({"indicator": "Unemployment rate", "by_country": out})


if __name__ == "__main__":
    main()
