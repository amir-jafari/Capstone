META = dict(
    name="Frankfurter (ECB exchange rates)",
    domain="Finance & Economics",
    docs="https://frankfurter.dev/",
    auth="None",
    signup="No registration, no limits.",
    provides=["Daily reference FX rates published by the European Central Bank (~30 currencies)",
              "Historical rates back to 1999 and time-series ranges"],
    ideas="Exchange-rate forecasting, currency-crisis analysis, converting multi-country price data.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    latest = get("https://api.frankfurter.dev/v1/latest", params={"base": "USD", "symbols": "EUR,GBP,JPY,CNY,INR"})
    series = get("https://api.frankfurter.dev/v1/2026-09-01..2026-09-05", params={"base": "USD", "symbols": "EUR"})
    show({"latest": latest, "usd_eur_series": series["rates"]})


if __name__ == "__main__":
    main()
