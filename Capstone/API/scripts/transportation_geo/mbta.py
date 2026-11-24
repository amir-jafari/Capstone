META = dict(
    name="MBTA V3 API (Boston transit)",
    domain="Transportation & Geospatial",
    docs="https://api-v3.mbta.com/docs/swagger/index.html",
    auth="None (20 req/min) / free key (1,000 req/min, header `x-api-key`)",
    env="MBTA_API_KEY (optional)",
    signup="Works keyless at low volume. Register at https://api-v3.mbta.com/register for a key. A good example of a "
           "modern JSON:API transit API (WMATA's needs a key; GTFS-Realtime feeds exist for most US agencies).",
    provides=["Real-time predictions and vehicle positions for subway, bus, commuter rail, ferry",
              "Routes, stops, schedules, service alerts", "include= joins related resources in one call"],
    ideas="Transit reliability (predicted vs. scheduled), headway analysis, alert text mining.",
    endpoint="GET https://api-v3.mbta.com/{resource}",
    params=[
        ("resource", "predictions | vehicles | schedules | alerts | routes | stops | trips", "predictions"),
        ("filter[stop] / filter[route]", "Filter by stop or route ID (comma lists)", "filter[stop]=place-pktrm"),
        ("filter[route_type]", "0 light rail, 1 subway, 2 commuter rail, 3 bus, 4 ferry", "1"),
        ("include", "Embed related resources", "route,trip"),
        ("sort", "Attribute to sort by (prefix - for descending)", "arrival_time"),
        ("page[limit] / page[offset]", "Paging", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api-v3.mbta.com/predictions", params={
        "filter[stop]": "place-pktrm", "sort": "arrival_time", "page[limit]": 12, "include": "trip"})
    trips = {t["id"]: t["attributes"]["headsign"] for t in d.get("included", []) if t["type"] == "trip"}
    show([{"route": p["relationships"]["route"]["data"]["id"],
           "headsign": trips.get(p["relationships"]["trip"]["data"]["id"]),
           "arrival_time": p["attributes"]["arrival_time"],  # null at a trip's first stop
           "departure_time": p["attributes"]["departure_time"]}
          for p in d["data"] if p["attributes"]["arrival_time"] or p["attributes"]["departure_time"]],  # drop skipped stops
         max_items=6)


if __name__ == "__main__":
    main()
