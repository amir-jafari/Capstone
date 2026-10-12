META = dict(
    name="U.S. Treasury daily yield curve rates",
    domain="Finance & Economics",
    docs="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve (the site may block scripted page views; the CSV works)",
    auth="None (CSV / XML feed)",
    signup="No registration. One CSV per year and rate type; also an XML (OData-style) feed. The par yield curve is "
           "the standard source for term-structure work.",
    provides=["Daily par yield curve (1 month to 30 years) since 1990", "Real (TIPS) yield curve, bill rates, long-term rates",
              "Yield-curve spreads (e.g. 10y-2y, 10y-3m) for recession signals"],
    ideas="Yield-curve inversion as a recession predictor, Nelson-Siegel term-structure modeling, rates vs. Fed policy.",
    endpoint="GET https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all",
    params=[
        ("year", "Calendar year in the path (and field_tdr_date_value)", "2026"),
        ("type", "daily_treasury_yield_curve | daily_treasury_real_yield_curve | daily_treasury_bill_rates | daily_treasury_long_term_rate", "daily_treasury_yield_curve"),
        ("field_tdr_date_value", "Year again (or YYYYMM for one month)", "2026"),
        ("_format", "csv", "csv"),
        ("columns", "Date, 1 Mo, 1.5 Month, 2 Mo, 3 Mo, 4 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr", "—"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2026/all"


def main():
    text = get(URL, params={"type": "daily_treasury_yield_curve", "field_tdr_date_value": 2026, "page": "",
                            "_format": "csv"}, as_json=False)
    rows = list(csv.DictReader(io.StringIO(text)))  # newest first
    latest = rows[0]
    show({"date": latest["Date"], "curve_pct": {k: float(v) for k, v in latest.items() if k != "Date" and v},
          "spread_10y_minus_2y": round(float(latest["10 Yr"]) - float(latest["2 Yr"]), 2),
          "spread_10y_minus_3m": round(float(latest["10 Yr"]) - float(latest["3 Mo"]), 2),
          "trading_days_in_2026": len(rows)}, max_items=15)


if __name__ == "__main__":
    main()
