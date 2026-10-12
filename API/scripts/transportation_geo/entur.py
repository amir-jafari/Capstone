META = dict(
    name="Entur (Norway national journey planner, GraphQL)",
    domain="Transportation & Geospatial",
    docs="https://developer.entur.org/",
    auth="None (ET-Client-Name header required)",
    signup="No registration, but every request must send an ET-Client-Name header like 'company-application'. All "
           "Norwegian public transport (trains, buses, ferries, flights) in one GraphQL API, plus a geocoder.",
    provides=["Trip planning with legs, modes, lines, real-time departures and delays",
              "Stop places, quays, departures boards; national stop registry (NSR) IDs", "Geocoder for stops and addresses"],
    ideas="Rural accessibility by public transport, multimodal trip analysis, real-time reliability studies.",
    endpoint="POST https://api.entur.io/journey-planner/v3/graphql  ·  GET https://api.entur.io/geocoder/v1/autocomplete",
    params=[
        ("geocoder: text / layers / size", "Find stop IDs (layers=venue for stops)", "text=Oslo S&layers=venue"),
        ("trip(from:{place:}, to:{place:})", "Trip patterns between NSR stop IDs", "NSR:StopPlace:59872 -> NSR:StopPlace:59983"),
        ("numTripPatterns / dateTime", "Number of alternatives; departure time (ISO)", "3"),
        ("stopPlace(id:) { estimatedCalls }", "Live departures from a stop", "—"),
        ("ET-Client-Name (header)", "Required identification", "gwu-capstone-catalog"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, get, show

HDR = {"ET-Client-Name": "gwu-capstone-catalog", "User-Agent": USER_AGENT}


def stop_id(name):
    f = get("https://api.entur.io/geocoder/v1/autocomplete", headers=HDR,
            params={"text": name, "layers": "venue", "size": 1})["features"][0]["properties"]
    return f["id"]


def main():
    a, b = stop_id("Oslo S"), stop_id("Bergen stasjon")
    q = f'{{ trip(from: {{place: "{a}"}}, to: {{place: "{b}"}}, numTripPatterns: 3) '
    q += '{ tripPatterns { expectedStartTime duration legs { mode line { publicCode name } } } } }'
    r = requests.post("https://api.entur.io/journey-planner/v3/graphql", json={"query": q}, headers=HDR, timeout=60)
    r.raise_for_status()
    trips = r.json()["data"]["trip"]["tripPatterns"]
    show({"from": a, "to": b, "trips": [
        {"departs": t["expectedStartTime"], "duration_h": round(t["duration"] / 3600, 1),
         "legs": [f'{l["mode"]} {(l.get("line") or {}).get("publicCode") or ""}'.strip() for l in t["legs"]]} for t in trips]})


if __name__ == "__main__":
    main()
