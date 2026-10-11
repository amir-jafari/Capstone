META = dict(
    name="US Bureau of Labor Statistics (BLS)",
    domain="Finance & Economics",
    docs="https://www.bls.gov/developers/",
    auth="None for v1; free key for v2",
    env="BLS_API_KEY (optional)",
    signup="v1 works keyless (25 queries/day, 10 years). Register at https://data.bls.gov/registrationEngine/ "
           "for v2: 500 queries/day, 20 years, 50 series per request, net/percent changes.",
    provides=["CPI (inflation), unemployment, employment by industry (CES), wages, productivity",
              "Local area unemployment (LAUS) for states/counties/metros", "Producer prices, JOLTS"],
    ideas="Labor-market analysis, inflation forecasting, regional employment shocks.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.bls.gov/publicAPI/v1/timeseries/data/CUUR0000SA0")  # CPI-U all items
    s = d["Results"]["series"][0]
    show({"series": s["seriesID"] + " (CPI-U, all items, 1982-84=100)",
          "data": [{"period": f'{x["year"]}-{x["periodName"]}', "value": x["value"]} for x in s["data"][:6]]})


if __name__ == "__main__":
    main()
