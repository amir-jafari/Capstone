META = dict(
    name="FRED (Federal Reserve Economic Data)",
    domain="Finance & Economics",
    docs="https://fred.stlouisfed.org/docs/api/fred/",
    auth="API key (query `api_key`)",
    env="FRED_API_KEY",
    signup="Create a free account at https://fredaccount.stlouisfed.org -> 'API Keys' -> Request API Key "
           "(describe use, e.g. 'GWU capstone'). 120 req/min.",
    provides=["800,000+ US & international time series: GDP, CPI, unemployment, rates, housing, M2...",
              "Series search, release calendars, vintage (real-time) data via ALFRED"],
    ideas="Macro forecasting, recession prediction, nowcasting, economic regime detection.",
    sample="""{
  "series": "UNRATE (Unemployment Rate, %)",
  "observations": [
    {"date": "2026-04-01", "value": "4.2"},
    {"date": "2026-05-01", "value": "4.2"},
    {"date": "2026-06-01", "value": "4.1"},
    {"date": "2026-07-01", "value": "4.2"},
    {"date": "2026-08-01", "value": "4.3"}
  ]
}""",
    endpoint="GET https://api.stlouisfed.org/fred/series/observations",
    params=[
        ("series_id", "Series code (UNRATE, GDP, CPIAUCSL, FEDFUNDS, DGS10, ...)", "UNRATE"),
        ("api_key / file_type", "Your key; file_type json | xml", "<FRED_API_KEY> / json"),
        ("observation_start / observation_end", "Date range YYYY-MM-DD", "2000-01-01"),
        ("frequency", "Aggregate to d | w | m | q | sa | a", "q"),
        ("aggregation_method", "avg | sum | eop (end of period)", "avg"),
        ("units", "lin | chg | ch1 | pch | pc1 | pca | log (levels, changes, % changes)", "pc1"),
        ("sort_order / limit / offset", "asc | desc; max 100,000 rows", "desc / 5"),
        ("/fred/series/search?search_text=", "Sibling endpoint: find series IDs by keyword", "search_text=housing starts"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    obs = get("https://api.stlouisfed.org/fred/series/observations", params={
        "series_id": "UNRATE", "api_key": key("FRED_API_KEY"), "file_type": "json",
        "sort_order": "desc", "limit": 5})["observations"]
    show({"series": "UNRATE (Unemployment Rate, %)",
          "observations": [{"date": o["date"], "value": o["value"]} for o in reversed(obs)]})


if __name__ == "__main__":
    main()
