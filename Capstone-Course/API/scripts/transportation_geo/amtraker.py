META = dict(
    name="Amtraker (live Amtrak train tracking, unofficial)",
    domain="Transportation & Geospatial",
    docs="https://github.com/piemadd/amtrak",
    auth="None",
    signup="No registration. Community-run API that decodes Amtrak's own train-tracking feed; unofficial, so endpoints "
           "can change. Please cache rather than poll aggressively.",
    provides=["Every active Amtrak (and VIA Rail / Brightline) train: route, train number, position, speed, heading",
              "Station-by-station schedule with scheduled vs. actual/estimated arrival and departure times",
              "Station list with coordinates and time zones"],
    ideas="On-time performance by route, delay propagation along a line, intercity rail vs. air travel times.",
    endpoint="GET https://api-v3.amtraker.com/v3/{trains | trains/{trainNum} | stations | stations/{code}}",
    params=[
        ("trains", "All active trains keyed by train number", "—"),
        ("trains/{trainNum}", "One train (may return several runs on different dates)", "trains/171"),
        ("stations / stations/{code}", "Station list or one station with upcoming trains", "stations/WAS"),
        ("train fields", "routeName, trainNum, trainState, lat, lon, velocity, heading, stations[] with schArr/arr", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    trains = [t for runs in get("https://api-v3.amtraker.com/v3/trains").values() for t in runs]
    active = [t for t in trains if t.get("trainState") == "Active"]
    nec = [t for t in active if t["routeName"] in ("Northeast Regional", "Acela")]
    show({"trains_in_feed": len(trains), "active": len(active),
          "busiest_routes": Counter(t["routeName"] for t in active).most_common(5),
          "northeast_corridor_now": [{"route": t["routeName"], "train": t["trainNum"], "speed_mph": round(t.get("velocity") or 0),
                                      "heading": t.get("heading"), "event": t.get("eventName")} for t in nec[:4]]})


if __name__ == "__main__":
    main()
