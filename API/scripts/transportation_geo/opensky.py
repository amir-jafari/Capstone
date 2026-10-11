META = dict(
    name="OpenSky Network (live flights)",
    domain="Transportation & Geospatial",
    docs="https://openskynetwork.github.io/opensky-api/rest.html",
    auth="None (anonymous: 400 credits/day) / OAuth client for more",
    signup="Anonymous works for live state vectors. Register at https://opensky-network.org for 4,000 credits/day "
           "and historical flights (create an API client under Account -> API Client).",
    provides=["Live aircraft state vectors: callsign, origin country, lat/lon, altitude, velocity, heading",
              "Arrivals/departures by airport, flight tracks (authenticated)"],
    ideas="Air-traffic pattern mining, emissions estimation, delay propagation.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://opensky-network.org/api/states/all",  # bounding box around Washington DC
            params={"lamin": 38.6, "lomin": -77.6, "lamax": 39.2, "lomax": -76.6})
    states = d.get("states") or []
    show({"aircraft_in_box": len(states), "sample": [
        {"callsign": s[1].strip(), "origin_country": s[2], "lat": s[6], "lon": s[5], "baro_alt_m": s[7],
         "velocity_mps": s[9]} for s in states]})


if __name__ == "__main__":
    main()
