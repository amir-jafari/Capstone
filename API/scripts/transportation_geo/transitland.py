META = dict(
    name="Transitland (GTFS transit data worldwide)",
    domain="Transportation & Geospatial",
    docs="https://www.transit.land/documentation/rest-api/",
    auth="API key (header `apikey`)",
    env="TRANSITLAND_API_KEY",
    signup="Sign up for a free key at https://www.transit.land/documentation#signing-up-for-an-api-key "
           "(free tier for non-commercial use). Aggregates 2,500+ GTFS feeds.",
    provides=["Transit agencies, routes, stops and schedules from GTFS feeds worldwide in one API",
              "Stop departures, route shapes (GeoJSON), feed versions over time", "Search near a point or inside a bounding box"],
    ideas="Transit-access scores for neighborhoods, comparing service levels across cities, service-change history.",
    endpoint="GET https://transit.land/api/v2/rest/{resource}",
    params=[
        ("resource", "agencies | routes | stops | feeds | stops/{key}/departures", "routes"),
        ("search", "Text search on names", "metro"),
        ("lat / lon / radius", "Near a point (radius in metres)", "38.8997 / -77.0486 / 500"),
        ("bbox", "min_lon,min_lat,max_lon,max_lat", "-77.12,38.80,-76.91,38.99"),
        ("route_type", "GTFS route type: 1 subway, 3 bus, ...", "1"),
        ("limit / after", "Paging (cursor in meta.after)", "5"),
        ("apikey (header)", "Your key", "<TRANSITLAND_API_KEY>"),
    ],
    sample="""[
  {"route": "RD", "long_name": "Red Line", "agency": "Washington Metropolitan Area Transit Authority", "route_type": 1},
  {"route": "BL", "long_name": "Blue Line", "agency": "Washington Metropolitan Area Transit Authority", "route_type": 1}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://transit.land/api/v2/rest/routes", headers={"apikey": key("TRANSITLAND_API_KEY")},
            params={"lat": 38.8997, "lon": -77.0486, "radius": 1000, "route_type": 1, "limit": 5})
    show([{"route": r["route_short_name"], "long_name": r["route_long_name"], "agency": r["agency"]["agency_name"],
           "route_type": r["route_type"]} for r in d["routes"]])


if __name__ == "__main__":
    main()
