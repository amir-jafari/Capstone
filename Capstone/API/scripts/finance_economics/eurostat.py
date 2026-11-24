META = dict(
    name="Eurostat API",
    domain="Finance & Economics",
    docs="https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started/api",
    auth="None",
    signup="No registration. Official EU statistics (~7,000 datasets). Note: in 2026 Eurostat moved HICP inflation to the "
           "new ECOICOP v2 classification - use prc_hicp_minr (dimension coicop18); the old prc_hicp_manr stops in Dec 2025.",
    provides=["Inflation (HICP), GDP and national accounts, unemployment, wages, house prices for EU countries and regions",
              "Population, migration, tourism, energy, environment, trade, regional (NUTS) statistics",
              "JSON-stat responses with labels; catalogue (table of contents) of all datasets"],
    ideas="Cross-country EU panels, regional convergence (NUTS 2), inflation decomposition by product.",
    endpoint="GET https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}",
    params=[
        ("dataset", "Dataset code: prc_hicp_minr (HICP), une_rt_m (unemployment), nama_10_gdp (GDP), prc_hpi_q (house prices)", "prc_hicp_minr"),
        ("dimension filters", "Any dimension as a parameter, repeatable: geo, unit, coicop18, sex, age, ...", "geo=EA&unit=RCH_A&coicop18=TOTAL"),
        ("lastTimePeriod / sinceTimePeriod / time", "Time filters", "lastTimePeriod=6"),
        ("lang", "en | fr | de", "en"),
        ("/catalogue/toc/txt", "Table of contents with dataset codes and update dates", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_minr", params=[
        ("geo", "EA"), ("geo", "DE"), ("geo", "FR"), ("geo", "IT"), ("geo", "ES"), ("unit", "RCH_A"),
        ("coicop18", "TOTAL"), ("lastTimePeriod", 3)])
    geos = d["dimension"]["geo"]["category"]["index"]
    times = d["dimension"]["time"]["category"]["index"]
    nt = len(times)
    # JSON-stat: values are a flat dict keyed by position = geo_index * n_times + time_index
    show({"series": d["label"] + " - annual rate of change, %", "data": {
        g: {t: d["value"].get(str(gi * nt + ti)) for t, ti in times.items()} for g, gi in geos.items()}})


if __name__ == "__main__":
    main()
