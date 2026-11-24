META = dict(
    name="US Energy Information Administration (EIA) v2",
    domain="Engineering, Energy & Tech",
    docs="https://www.eia.gov/opendata/documentation.php",
    auth="API key (query `api_key`)",
    env="EIA_API_KEY",
    signup="Register at https://www.eia.gov/opendata/register.php (key emailed). Free, ~5,000 req/hour.",
    provides=["Hourly electricity demand & generation by fuel for every US balancing authority",
              "Electricity prices by state/sector, petroleum, natural gas, coal, renewables, CO2 emissions",
              "Short-Term & Annual Energy Outlook forecasts"],
    ideas="Load forecasting, renewable-share analysis, grid-carbon-intensity modeling.",
    sample="""[
  {"period": "2026-10-10T23", "respondent": "PJM", "type-name": "Demand", "value": 92841, "value-units": "megawatthours"},
  {"period": "2026-10-10T22", "respondent": "PJM", "type-name": "Demand", "value": 95310, "value-units": "megawatthours"}
]""",
    endpoint="GET https://api.eia.gov/v2/{route}/data/",
    params=[
        ("route", "Dataset route; GET /v2/{route} (no /data) to browse facets and children", "electricity/rto/region-data"),
        ("frequency", "hourly | local-hourly | daily | monthly | quarterly | annual (depends on route)", "hourly"),
        ("data[0]", "Data column(s) to return", "value"),
        ("facets[{name}][]", "Facet filters, e.g. respondent (balancing authority), type (D = demand, NG = net generation, DF = forecast)", "facets[respondent][]=PJM"),
        ("start / end", "Period range in the route's format", "2026-10-01T00"),
        ("sort[0][column] / sort[0][direction]", "Sort", "period / desc"),
        ("length / offset", "Max 5,000 rows per request", "5"),
        ("api_key", "Your key", "<EIA_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.eia.gov/v2/electricity/rto/region-data/data/", params={
        "api_key": key("EIA_API_KEY"), "frequency": "hourly", "data[0]": "value",
        "facets[respondent][]": "PJM", "facets[type][]": "D",
        "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 5})
    show([{k: r[k] for k in ("period", "respondent", "type-name", "value", "value-units")} for r in d["response"]["data"]])


if __name__ == "__main__":
    main()
