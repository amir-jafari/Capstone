META = dict(
    name="JPL Fireball (bolide) API",
    domain="Science & Research",
    docs="https://ssd-api.jpl.nasa.gov/doc/fireball.html",
    auth="None",
    signup="No registration. Bright meteors detected by U.S. government sensors, reported by CNEOS since 1988.",
    provides=["Each fireball: date/time, latitude/longitude, altitude, velocity, radiated energy, estimated impact energy (kt TNT)"],
    ideas="Spatial/temporal distribution of atmospheric impacts, energy-frequency power laws.",
    endpoint="GET https://ssd-api.jpl.nasa.gov/fireball.api",
    params=[
        ("date-min / date-max", "Date range", "2025-01-01"),
        ("energy-min / impact-e-min", "Minimum radiated energy (J x10^10) / impact energy (kt)", "impact-e-min=1"),
        ("req-loc / req-alt / req-vel", "true -> only events with location / altitude / velocity", "req-loc=true"),
        ("sort", "date | energy | impact-e | alt | vel (prefix - for descending)", "-impact-e"),
        ("limit", "Max rows", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ssd-api.jpl.nasa.gov/fireball.api", params={"date-min": "2015-01-01", "req-loc": "true",
                                                                  "sort": "-impact-e", "limit": 5})
    show({"largest_since_2015": [dict(zip(d["fields"], r)) for r in d["data"]]})


if __name__ == "__main__":
    main()
