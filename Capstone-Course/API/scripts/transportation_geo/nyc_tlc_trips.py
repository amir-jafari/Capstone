META = dict(
    name="NYC Taxi & Limousine Commission trip records",
    domain="Transportation & Geospatial",
    docs="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
    auth="None (Parquet downloads)",
    signup="No registration; no REST API - monthly Parquet files (read with pandas + pyarrow). Sizes per month: green taxi "
           "~1 MB, yellow taxi ~60 MB, high-volume for-hire (Uber/Lyft) ~490 MB. Data dictionaries on the docs page.",
    provides=["Every yellow/green taxi trip since 2009: pickup/dropoff time and taxi zone, distance, fare, tip, tolls, payment",
              "High-volume for-hire trips (Uber, Lyft, Via) since 2019 incl. driver pay and wait times",
              "Taxi-zone lookup table and shapefile"],
    ideas="Fare and tip prediction, demand forecasting by zone, taxi vs. rideshare competition, congestion-pricing effects.",
    endpoint="GET https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{YYYY-MM}.parquet",
    params=[
        ("type", "yellow | green | fhv | fhvhv", "green"),
        ("YYYY-MM", "Month (published with a ~2-month lag)", "2026-07"),
        ("misc/taxi_zone_lookup.csv", "LocationID -> borough, zone, service_zone", "—"),
        ("key columns (yellow/green)", "lpep/tpep_pickup_datetime, PULocationID, DOLocationID, trip_distance, fare_amount, tip_amount, payment_type", "—"),
    ],
)
import io

import pandas as pd
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

BASE = "https://d37ci6vzurychx.cloudfront.net"


def main():
    raw = requests.get(f"{BASE}/trip-data/green_tripdata_2026-07.parquet", headers={"User-Agent": USER_AGENT}, timeout=120)
    trips = pd.read_parquet(io.BytesIO(raw.content))
    zones = pd.read_csv(f"{BASE}/misc/taxi_zone_lookup.csv")
    card = trips[trips["payment_type"] == 1]  # 1 = credit card (cash tips are not recorded)
    top = (trips.merge(zones, left_on="PULocationID", right_on="LocationID")["Zone"].value_counts().head(4))
    show({"month": "2026-07 (green taxis)", "trips": len(trips),
          "median_distance_mi": round(trips["trip_distance"].median(), 2),
          "median_fare_usd": round(trips["fare_amount"].median(), 2),
          "card_tip_pct_of_fare": round(100 * card["tip_amount"].sum() / card["fare_amount"].sum(), 1),
          "top_pickup_zones": top.to_dict()})


if __name__ == "__main__":
    main()
