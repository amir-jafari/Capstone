META = dict(
    name="JPL Small-Body Close Approach & SBDB APIs",
    domain="Science & Research",
    docs="https://ssd-api.jpl.nasa.gov/doc/cad.html",
    auth="None",
    signup="No registration. Part of JPL's Solar System Dynamics APIs (also sbdb.api for orbits/physical data, "
           "sentry.api for impact risk, scout.api for new objects).",
    provides=["Close approaches of asteroids and comets to Earth (or other bodies): date, distance, velocity, size proxy (H)",
              "Small-Body Database: orbit elements, diameter, albedo, rotation, discovery", "Sentry impact-risk list"],
    ideas="Near-Earth object population statistics, size-frequency distributions, close-approach trend analysis.",
    endpoint="GET https://ssd-api.jpl.nasa.gov/{cad | sbdb | sentry}.api",
    params=[
        ("date-min / date-max", "Window (YYYY-MM-DD, or now / +60 days)", "now / +60"),
        ("dist-max", "Max distance in au (or LD for lunar distances, e.g. 1LD)", "1LD"),
        ("h-max", "Max absolute magnitude (smaller H = larger object)", "22"),
        ("body", "Earth (default) | Moon | Mars | ALL ...", "Earth"),
        ("sort", "date | dist | v-rel | h (prefix - for descending)", "dist"),
        ("sbdb.api?sstr=", "Small-Body Database record for one object (phys-par=1 adds size/albedo)", "sstr=Apophis&phys-par=1"),
        ("sentry.api", "Objects with non-zero impact probability", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ssd-api.jpl.nasa.gov/cad.api", params={"date-min": "now", "date-max": "+60", "dist-max": "1LD",
                                                             "sort": "dist"})
    rows = [dict(zip(d["fields"], r)) for r in d["data"]]
    apophis = get("https://ssd-api.jpl.nasa.gov/sbdb.api", params={"sstr": "Apophis", "phys-par": 1})
    phys = {p["name"]: p["value"] for p in apophis.get("phys_par", [])}
    show({"approaches_within_1_lunar_distance_next_60_days": d["count"], "closest": [
        {"object": r["des"], "date": r["cd"], "distance_lunar": round(float(r["dist"]) / 0.00257, 2),
         "velocity_km_s": round(float(r["v_rel"]), 1), "H": r["h"]} for r in rows[:5]],
          "apophis": {"name": apophis["object"]["fullname"], "diameter_km": phys.get("diameter"), "albedo": phys.get("albedo")}})


if __name__ == "__main__":
    main()
