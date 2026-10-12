META = dict(
    name="Chicago Taxi & Rideshare Trips",
    domain="Transportation & Geospatial",
    docs="https://data.cityofchicago.org/Transportation/Taxi-Trips-2024-/ajtu-isnz",
    auth="None (optional Socrata app token)",
    env="SOCRATA_APP_TOKEN (optional)",
    signup="No registration. Large tables: always filter by date and aggregate server-side ($select + $group) rather than "
           "pulling raw rows. Rideshare (TNP) trips are a separate dataset on the same portal.",
    provides=["Every taxi trip since 2024: start/end time (15-min), duration, miles, community areas, fare, tip, payment type, company",
              "Pickup/dropoff centroids (lat/lon), hashed taxi IDs", "Older trips (2013-2023) in a separate dataset"],
    ideas="Fare and tip prediction, demand forecasting by area/hour, taxi vs. rideshare competition.",
    endpoint="GET https://data.cityofchicago.org/resource/ajtu-isnz.json",
    params=[
        ("$where", "Filter rows - always bound trip_start_timestamp", "trip_start_timestamp between '2026-09-01T00:00:00' and '2026-09-08T00:00:00'"),
        ("$select + $group", "Server-side aggregation", "payment_type, count(*) as trips, avg(fare)"),
        ("$order / $limit", "Sort and limit", "trips DESC / 5"),
        ("columns", "trip_seconds, trip_miles, pickup_community_area, fare, tips, payment_type, company, ...", "—"),
        ("X-App-Token (header)", "Optional app token", "<SOCRATA_APP_TOKEN>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://data.cityofchicago.org/resource/ajtu-isnz.json", timeout=120, params={
        "$select": "payment_type, count(*) as trips, avg(fare) as avg_fare, avg(tips) as avg_tip, avg(trip_miles) as avg_miles",
        "$where": "trip_start_timestamp between '2026-09-01T00:00:00' and '2026-09-08T00:00:00'",
        "$group": "payment_type", "$order": "trips DESC", "$limit": 5})
    show([{k: (round(float(v), 2) if k.startswith("avg") else v) for k, v in r.items()} for r in d])


if __name__ == "__main__":
    main()
