META = dict(
    name="GBFS Bike-share Feeds (Capital Bikeshare and 1,000+ systems)",
    domain="Transportation & Geospatial",
    docs="https://github.com/MobilityData/gbfs (spec) · system list: https://github.com/MobilityData/gbfs/blob/master/systems.csv",
    auth="None",
    signup="No registration. GBFS is an open standard: every public bike/scooter-share system publishes the same JSON feeds. "
           "Trip-history CSVs for Capital Bikeshare: https://capitalbikeshare.com/system-data.",
    provides=["Real-time station status: bikes / e-bikes / docks available, renting/returning flags",
              "Station information: name, lat/lon, capacity", "Free-floating vehicles, pricing plans, system alerts"],
    ideas="Rebalancing / demand forecasting, equity of station access, weather effects on micromobility.",
    endpoint="GET https://gbfs.lyft.com/gbfs/2.3/dca-cabi/en/{feed}.json",
    params=[
        ("gbfs.json", "Auto-discovery: lists every feed URL for the system", "gbfs/2.3/dca-cabi/gbfs.json"),
        ("station_information", "Static: station_id, name, lat, lon, capacity", "station_information.json"),
        ("station_status", "Real time: num_bikes_available, num_ebikes_available, num_docks_available", "station_status.json"),
        ("free_bike_status / vehicle_status", "Dockless vehicles with location", "free_bike_status.json"),
        ("system_id", "Other systems: bay-wheels, cdc-divvy (Chicago), bkn (Citi Bike NYC)...", "dca-cabi"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://gbfs.lyft.com/gbfs/2.3/dca-cabi/en"


def main():
    info = {s["station_id"]: s for s in get(f"{BASE}/station_information.json")["data"]["stations"]}
    status = get(f"{BASE}/station_status.json")
    stations = status["data"]["stations"]
    top = sorted(stations, key=lambda s: -s["num_bikes_available"])[:5]
    show({"stations": len(stations), "bikes_available_total": sum(s["num_bikes_available"] for s in stations),
          "fullest_stations": [{"name": info[s["station_id"]]["name"], "bikes": s["num_bikes_available"],
                                "ebikes": s.get("num_ebikes_available"), "docks_free": s["num_docks_available"]}
                               for s in top if s["station_id"] in info]})


if __name__ == "__main__":
    main()
