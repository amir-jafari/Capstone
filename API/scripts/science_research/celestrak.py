META = dict(
    name="CelesTrak (satellite orbital elements)",
    domain="Science & Research",
    docs="https://celestrak.org/NORAD/documentation/gp-data-formats.php",
    auth="None",
    signup="No registration. Download each group at most once every 2 hours - a repeat request returns HTTP 403 "
           "until the data updates. Elements use the "
           "OMM format; propagate positions with the sgp4 Python package.",
    provides=["Current orbital elements (General Perturbations) for ~30,000 tracked objects: stations, Starlink, GPS, weather sats, debris",
              "Epoch, inclination, eccentricity, mean motion, NORAD ID, international designator",
              "SATCAT catalog: owner country, launch date, decay date, object type"],
    ideas="Satellite-constellation growth (Starlink), orbital-debris analysis, pass prediction with sgp4.",
    endpoint="GET https://celestrak.org/NORAD/elements/gp.php",
    params=[
        ("GROUP", "Named set: stations | starlink | gps-ops | weather | active | last-30-days | ...", "starlink"),
        ("CATNR / INTDES / NAME", "Single object by NORAD number, international designator, or name", "CATNR=25544 (ISS)"),
        ("FORMAT", "json | csv | xml | tle | 3le", "json"),
        ("satcat/records.php?GROUP=", "Sibling endpoint: catalog records (owner, launch, decay)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    starlink = get("https://celestrak.org/NORAD/elements/gp.php", params={"GROUP": "starlink", "FORMAT": "json"})
    iss = get("https://celestrak.org/NORAD/elements/gp.php", params={"CATNR": 25544, "FORMAT": "json"})[0]
    shells = {}
    for s in starlink:
        shells[round(s["INCLINATION"])] = shells.get(round(s["INCLINATION"]), 0) + 1
    show({"starlink_satellites_tracked": len(starlink),
          "starlink_by_inclination_deg": dict(sorted(shells.items(), key=lambda kv: -kv[1])[:5]),
          "iss": {k: iss[k] for k in ("OBJECT_NAME", "EPOCH", "INCLINATION", "MEAN_MOTION", "ECCENTRICITY")} |
                 {"orbital_period_min": round(1440 / iss["MEAN_MOTION"], 1)}})


if __name__ == "__main__":
    main()
