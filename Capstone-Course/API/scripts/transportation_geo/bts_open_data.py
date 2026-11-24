META = dict(
    name="Bureau of Transportation Statistics (BTS) Open Data",
    domain="Transportation & Geospatial",
    docs="https://data.bts.gov/ (Socrata SODA API; same query language as the CDC entry)",
    auth="None (optional Socrata app token)",
    env="SOCRATA_APP_TOKEN (optional)",
    signup="No registration. Every dataset page has an API button with its 4x4 ID. Airline on-time performance is a "
           "separate bulk download at https://www.transtats.bts.gov.",
    provides=["Border crossings by port, month and mode (trucks, buses, pedestrians, personal vehicles)",
              "Monthly Transportation Statistics: air/transit ridership, freight, fuel prices, safety",
              "Freight flows, transportation employment, pipeline and rail statistics"],
    ideas="Trade/mobility shocks at the border, transportation as an economic indicator, post-pandemic travel recovery.",
    endpoint="GET https://data.bts.gov/resource/{dataset-id}.json",
    params=[
        ("dataset-id", "4x4 ID from the dataset page", "keg4-3bc2 (Border Crossing Entry Data)"),
        ("$select / $where / $group / $order", "SQL-like query clauses (see the CDC entry)", "$select=port_name,sum(value)&$group=port_name"),
        ("column=value", "Equality filter", "measure=Trucks"),
        ("$limit / $offset", "Paging (default 1,000)", "5"),
        ("X-App-Token (header)", "Optional app token", "<SOCRATA_APP_TOKEN>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://data.bts.gov/resource/keg4-3bc2.json", params={
        "$select": "port_name,state,sum(value) as trucks", "measure": "Trucks", "border": "US-Mexico Border",
        "$where": "date >= '2025-01-01T00:00:00' AND date < '2026-01-01T00:00:00'",
        "$group": "port_name,state", "$order": "trucks DESC", "$limit": 5})
    show({"busiest_US_Mexico_truck_crossings_2025": d})


if __name__ == "__main__":
    main()
