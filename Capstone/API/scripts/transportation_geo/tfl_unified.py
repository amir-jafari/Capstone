META = dict(
    name="Transport for London Unified API",
    domain="Transportation & Geospatial",
    docs="https://api-portal.tfl.gov.uk/",
    auth="None (free app key raises limits)",
    env="TFL_APP_KEY (optional)",
    signup="Works without a key at low volume. Register at https://api-portal.tfl.gov.uk for a key (500 requests/minute). "
           "One of the richest city transit APIs in the world.",
    provides=["Live line status and disruptions for Tube, Overground, Elizabeth line, DLR, buses, trams",
              "Arrival predictions per stop, journey planner, Santander Cycles docks, road disruptions",
              "Air-quality forecasts, road disruptions, car parks"],
    ideas="Disruption pattern analysis, reliability by line, bike-share availability vs. weather and events.",
    endpoint="GET https://api.tfl.gov.uk/{resource}",
    params=[
        ("Line/Mode/{modes}/Status", "Status of every line in the modes (tube, dlr, overground, elizabeth-line, bus)", "tube,elizabeth-line"),
        ("StopPoint/{id}/Arrivals", "Live arrivals at a stop", "940GZZLUOXC (Oxford Circus)"),
        ("Journey/JourneyResults/{from}/to/{to}", "Journey planner (postcodes, stop IDs or lat,lon)", "SW1A1AA/to/E14 5AB"),
        ("BikePoint", "All cycle-hire docks with bikes/empty docks", "—"),
        ("app_key", "Optional key", "<TFL_APP_KEY>"),
    ],
)
import os
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    params = {"app_key": os.environ["TFL_APP_KEY"]} if os.getenv("TFL_APP_KEY") else None
    lines = get("https://api.tfl.gov.uk/Line/Mode/tube,elizabeth-line,dlr/Status", params=params)
    bikes = get("https://api.tfl.gov.uk/BikePoint", params=params)
    nbikes = sum(int(next((p["value"] for p in b["additionalProperties"] if p["key"] == "NbBikes"), 0)) for b in bikes)
    show({"line_status": {l["name"]: l["lineStatuses"][0]["statusSeverityDescription"] for l in lines},
          "status_counts": Counter(l["lineStatuses"][0]["statusSeverityDescription"] for l in lines).most_common(),
          "bike_docks": len(bikes), "bikes_available": nbikes}, max_items=15)


if __name__ == "__main__":
    main()
