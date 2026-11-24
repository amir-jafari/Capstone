META = dict(
    name="Swiss public transport API (transport.opendata.ch)",
    domain="Transportation & Geospatial",
    docs="https://transport.opendata.ch/docs.html",
    auth="None",
    signup="No registration. Max ~1,000 connection requests/day per IP. Covers all Swiss trains, trams, buses, boats.",
    provides=["Door-to-door connections: departure/arrival times, duration, transfers, platforms, delays, sections",
              "Live departure boards per station", "Station search by name or coordinates"],
    ideas="Comparing timetable connectivity across cities, delay analysis, multimodal travel-time accessibility.",
    endpoint="GET https://transport.opendata.ch/v1/{connections | stationboard | locations}",
    params=[
        ("from / to / via", "Stations or addresses", "Zürich HB / Genève"),
        ("date / time / isArrivalTime", "When to travel", "2026-10-20 / 08:00"),
        ("limit", "Number of connections (max 16)", "3"),
        ("transportations[]", "train | tram | bus | ship | cableway", "train"),
        ("stationboard?station=&limit=", "Departures from a station", "station=Bern&limit=10"),
        ("locations?query= / ?x=&y=", "Station search by name or coordinates", "query=Lausanne"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://transport.opendata.ch/v1/connections", params={"from": "Zürich HB", "to": "Genève", "limit": 3})
    board = get("https://transport.opendata.ch/v1/stationboard", params={"station": "Bern", "limit": 5})
    show({"zurich_to_geneva": [{"departure": c["from"]["departure"], "arrival": c["to"]["arrival"],
                                "duration": c["duration"], "transfers": c["transfers"],
                                "products": c["products"], "platform": c["from"].get("platform")} for c in d["connections"]],
          "bern_departures": [{"train": f'{s["category"]} {s["number"]}', "to": s["to"],
                               "time": s["stop"]["departure"], "delay_min": s["stop"].get("delay")} for s in board["stationboard"]]})


if __name__ == "__main__":
    main()
