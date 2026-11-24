META = dict(
    name="WMATA (DC Metro) API",
    domain="Transportation & Geospatial",
    docs="https://developer.wmata.com/apis",
    auth="API key (header `api_key`)",
    env="WMATA_API_KEY",
    signup="Sign up at https://developer.wmata.com -> Products -> subscribe to 'Default Tier' -> Profile shows "
           "primary key. 10 calls/sec, 50,000/day. GTFS & GTFS-RT feeds included.",
    provides=["Real-time rail predictions (next trains) per station", "Bus positions & predictions",
              "Rail incidents, elevator/escalator outages", "Station & route metadata, GTFS"],
    ideas="Transit reliability analysis, ridership/delay prediction, accessibility (elevator outage) studies.",
    sample="""[
  {"Line": "RD", "Destination": "Glenmont", "Min": "3", "Car": "8"},
  {"Line": "RD", "Destination": "Shady Grove", "Min": "5", "Car": "6"},
  {"Line": "RD", "Destination": "Glenmont", "Min": "11", "Car": "8"}
]""",
    endpoint="GET https://api.wmata.com/StationPrediction.svc/json/GetPrediction/{StationCodes}",
    params=[
        ("StationCodes", "Comma list of station codes, or All", "A01 (Metro Center)"),
        ("/Rail.svc/json/jStations?LineCode=", "Sibling endpoint: stations (RD, BL, OR, SV, GR, YL)", "LineCode=RD"),
        ("/Incidents.svc/json/Incidents · /ElevatorIncidents", "Sibling endpoints: rail disruptions, elevator/escalator outages", "—"),
        ("/NextBusService.svc/json/jPredictions?StopID=", "Sibling endpoint: bus arrival predictions", "StopID=1001195"),
        ("/Bus.svc/json/jBusPositions?RouteID=&Lat=&Lon=&Radius=", "Sibling endpoint: live bus locations", "RouteID=30N"),
        ("api_key (header)", "Your primary key", "<WMATA_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.wmata.com/StationPrediction.svc/json/GetPrediction/A01",  # Metro Center
            headers={"api_key": key("WMATA_API_KEY")})
    show([{k: t[k] for k in ("Line", "Destination", "Min", "Car")} for t in d["Trains"]])


if __name__ == "__main__":
    main()
