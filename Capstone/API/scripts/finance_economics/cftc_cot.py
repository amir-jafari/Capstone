META = dict(
    name="CFTC Commitments of Traders (Public Reporting Environment)",
    domain="Finance & Economics",
    docs="https://publicreporting.cftc.gov/stories/s/r4w3-av2u",
    auth="None (optional Socrata app token)",
    env="SOCRATA_APP_TOKEN (optional)",
    signup="No registration. Socrata SODA API (same query language as the CDC/Chicago entries). Weekly reports, every "
           "Friday, positions as of Tuesday.",
    provides=["Futures positioning by trader type for ~500 markets (commodities, rates, FX, equity indices, crypto)",
              "Legacy (commercial vs. non-commercial), disaggregated (producers, swap dealers, managed money) and TFF reports",
              "Open interest, long/short positions, changes, concentration ratios since 1986"],
    ideas="Speculator positioning as a price signal, hedging pressure studies, sentiment indices for commodities.",
    endpoint="GET https://publicreporting.cftc.gov/resource/{dataset}.json",
    params=[
        ("dataset", "6dca-aqww Legacy futures only | 72hh-3qpy Disaggregated futures | gpe5-46if Traders in Financial Futures", "6dca-aqww"),
        ("$where", "Filter, e.g. by market name and date", "commodity_name='CRUDE OIL'"),
        ("$order / $limit", "Sort and limit", "report_date_as_yyyy_mm_dd DESC / 5"),
        ("key columns", "market_and_exchange_names, report_date_as_yyyy_mm_dd, open_interest_all, noncomm_positions_long_all, noncomm_positions_short_all", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    rows = get("https://publicreporting.cftc.gov/resource/6dca-aqww.json", params={
        "$where": "market_and_exchange_names like 'WTI-PHYSICAL%'", "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 4})
    show({"market": rows[0]["market_and_exchange_names"], "weeks": [
        {"date": r["report_date_as_yyyy_mm_dd"][:10], "open_interest": int(r["open_interest_all"]),
         "noncommercial_net_long": int(r["noncomm_positions_long_all"]) - int(r["noncomm_positions_short_all"])}
        for r in rows]})


if __name__ == "__main__":
    main()
