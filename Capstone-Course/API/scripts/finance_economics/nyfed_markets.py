META = dict(
    name="New York Fed Markets Data API",
    domain="Finance & Economics",
    docs="https://markets.newyorkfed.org/static/docs/markets-api.html",
    auth="None",
    signup="No registration. Official source of the U.S. reference rates (SOFR, EFFR) and Fed open-market operations.",
    provides=["Reference rates: SOFR (with percentiles and volume), EFFR, OBFR, TGCR, BGCR, SOFR averages and index",
              "Repo and reverse-repo operations, Treasury and agency MBS purchases/sales",
              "System Open Market Account (SOMA) holdings; primary dealer statistics"],
    ideas="Money-market stress analysis, quantitative tightening (SOMA runoff), rate-spread dynamics around policy meetings.",
    endpoint="GET https://markets.newyorkfed.org/api/{path}.json",
    params=[
        ("rates/secured/sofr/last/{n}", "Last n SOFR fixings", "last/5"),
        ("rates/unsecured/effr/search?startDate=&endDate=", "EFFR over a date range", "startDate=2026-09-01&endDate=2026-09-30"),
        ("rates/all/latest", "Latest value of every reference rate", "—"),
        ("rp/reverserepo/propositions/search?startDate=&endDate=", "Overnight reverse-repo operations", "—"),
        ("soma/summary", "SOMA holdings over time by security type", "—"),
        ("format", ".json | .csv | .xml suffix", ".json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://markets.newyorkfed.org/api"


def main():
    latest = get(f"{BASE}/rates/all/latest.json")["refRates"]
    sofr = get(f"{BASE}/rates/secured/sofr/last/5.json")["refRates"]
    soma = get(f"{BASE}/soma/summary.json")["soma"]["summary"]
    # Rates carry percentRate; SOFRAI (the SOFR averages/index row) carries index and 30/90/180-day averages instead
    show({"latest_reference_rates_pct": {r["type"]: f'{r["percentRate"]} ({r["effectiveDate"]})' for r in latest
                                         if "percentRate" in r},
          "sofr_last_5": [{"date": r["effectiveDate"], "rate": r["percentRate"], "volume_bn": r["volumeInBillions"]} for r in sofr],
          "soma_total_holdings_bn": {s["asOfDate"]: round(float(s["total"]) / 1e9) for s in soma[-3:]}})


if __name__ == "__main__":
    main()
