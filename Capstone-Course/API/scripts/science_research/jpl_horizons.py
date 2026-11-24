META = dict(
    name="JPL Horizons (ephemerides)",
    domain="Science & Research",
    docs="https://ssd-api.jpl.nasa.gov/doc/horizons.html",
    auth="None",
    signup="No registration. The authoritative source for positions of planets, moons, asteroids, comets and spacecraft. "
           "The response is a text report inside JSON; parse the block between $$SOE and $$EOE.",
    provides=["Where any solar-system body (or spacecraft) is at any time: RA/Dec, distance, magnitude, phase, rise/set",
              "State vectors (x, y, z, vx, vy, vz) for orbital mechanics", "Physical data for planets and moons"],
    ideas="Orbit visualization, observation planning, validating orbital-mechanics simulations against real ephemerides.",
    endpoint="GET https://ssd.jpl.nasa.gov/api/horizons.api",
    params=[
        ("COMMAND", "Target in quotes: 499 Mars, 301 Moon, -125544 ISS, 'Ceres;' (URL-encode the ; when using curl - requests does it)", "'499'"),
        ("EPHEM_TYPE", "OBSERVER | VECTORS | ELEMENTS", "OBSERVER"),
        ("CENTER", "Observer location: 500@399 Earth geocenter, coord@399 with SITE_COORD, 500@10 Sun", "'500@399'"),
        ("START_TIME / STOP_TIME / STEP_SIZE", "Time span and step", "'2026-10-12' / '2026-10-15' / '1d'"),
        ("QUANTITIES (observer)", "1 RA/Dec, 9 magnitude, 20 range, 4 alt/az, ...", "'1,9,20'"),
        ("format", "json | text", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ssd.jpl.nasa.gov/api/horizons.api", params={
        "format": "json", "COMMAND": "'499'", "EPHEM_TYPE": "OBSERVER", "CENTER": "'500@399'",
        "START_TIME": "'2026-10-12'", "STOP_TIME": "'2026-10-15'", "STEP_SIZE": "'1d'", "QUANTITIES": "'1,9,20'"})
    text = d["result"]
    table = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    header = [l for l in text.splitlines() if "Date__(UT)__HR:MN" in l][0]
    show({"target": "Mars as seen from Earth's center", "columns": " ".join(header.split()),
          "rows": [" ".join(r.split()) for r in table]})


if __name__ == "__main__":
    main()
