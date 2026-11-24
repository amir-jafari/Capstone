META = dict(
    name="OSRM (routing on OpenStreetMap)",
    domain="Transportation & Geospatial",
    docs="https://project-osrm.org/docs/v5.24.0/api/",
    auth="None",
    signup="The public demo server allows light use only (~1 request/second, no bulk jobs). For real projects run your "
           "own OSRM with Docker on an OSM extract (car, bike or foot profiles).",
    provides=["Fastest route between points: distance, duration, turn-by-turn steps, geometry",
              "Travel-time / distance matrices between many points (table service)",
              "Snap GPS traces to roads (match), nearest road, travelling-salesman trips"],
    ideas="Accessibility measures (travel time to hospitals/jobs), commute-time features for housing models.",
    endpoint="GET https://router.project-osrm.org/{service}/v1/{profile}/{lon,lat;lon,lat;...}",
    params=[
        ("service", "route | table | nearest | match | trip", "route"),
        ("profile", "driving on the demo server (car | bike | foot on self-hosted)", "driving"),
        ("coordinates", "lon,lat pairs separated by ; (note: longitude first)", "-77.0486,38.8997;-77.0091,38.8899"),
        ("overview", "full | simplified | false (route geometry)", "false"),
        ("steps", "true -> turn-by-turn instructions", "true"),
        ("annotations (table)", "duration | distance | duration,distance", "duration,distance"),
        ("sources / destinations (table)", "Indexes of coordinates to use as origins / destinations", "sources=0"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

PLACES = {"GWU": (-77.0486, 38.8997), "Union Station": (-77.0063, 38.8973),
          "Reagan Airport": (-77.0402, 38.8512), "Dulles Airport": (-77.4565, 38.9531)}


def main():
    coords = ";".join(f"{lon},{lat}" for lon, lat in PLACES.values())
    d = get(f"https://router.project-osrm.org/table/v1/driving/{coords}",
            params={"sources": 0, "annotations": "duration,distance"})
    names = list(PLACES)
    show([{"from": "GWU", "to": names[i], "drive_min": round(d["durations"][0][i] / 60, 1),
           "km": round(d["distances"][0][i] / 1000, 1)} for i in range(1, len(names))])


if __name__ == "__main__":
    main()
