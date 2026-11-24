META = dict(
    name="MTA Subway Hourly Ridership (NY Open Data)",
    domain="Transportation & Geospatial",
    docs="https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-2025/5wq4-mkjj",
    auth="None (optional Socrata app token)",
    env="SOCRATA_APP_TOKEN (optional)",
    signup="No registration. Socrata SODA API (same query language as the CDC and Chicago entries). Earlier years are "
           "separate datasets (wujg-7c2s for 2020-2024, t69i-h2me for 2017-2019).",
    provides=["Hourly entries by station complex, payment method (OMNY/MetroCard), fare class, with coordinates",
              "Borough and station IDs for joining GTFS and Census data", "Hundreds of millions of rows - aggregate server-side"],
    ideas="Ridership recovery and seasonality, OMNY adoption by neighborhood, congestion-pricing effects.",
    endpoint="GET https://data.ny.gov/resource/5wq4-mkjj.json",
    params=[
        ("$select + $group", "Aggregate on the server", "station_complex, sum(ridership) as riders"),
        ("$where", "Always bound transit_timestamp", "transit_timestamp between '2026-09-01T00:00:00' and '2026-09-08T00:00:00'"),
        ("columns", "transit_timestamp, station_complex, borough, payment_method, fare_class_category, ridership, transfers, latitude, longitude", "—"),
        ("$order / $limit", "Sort and limit", "riders DESC / 5"),
        ("X-App-Token (header)", "Optional app token", "<SOCRATA_APP_TOKEN>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

URL = "https://data.ny.gov/resource/5wq4-mkjj.json"
WEEK = "transit_timestamp between '2026-09-01T00:00:00' and '2026-09-08T00:00:00'"


def main():
    top = get(URL, timeout=120, params={"$select": "station_complex, borough, sum(ridership) as riders",
                                       "$where": WEEK, "$group": "station_complex, borough",
                                       "$order": "riders DESC", "$limit": 5})
    pay = get(URL, timeout=120, params={"$select": "payment_method, sum(ridership) as riders", "$where": WEEK,
                                       "$group": "payment_method"})
    show({"week": "2026-09-01 to 2026-09-07",
          "busiest_stations": [{"station": r["station_complex"], "borough": r["borough"],
                                "riders": round(float(r["riders"]))} for r in top],
          "by_payment_method": {r["payment_method"]: round(float(r["riders"])) for r in pay}})


if __name__ == "__main__":
    main()
