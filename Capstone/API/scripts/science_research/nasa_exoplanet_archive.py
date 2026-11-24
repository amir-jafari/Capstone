META = dict(
    name="NASA Exoplanet Archive (TAP)",
    domain="Science & Research",
    docs="https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html",
    auth="None",
    signup="No registration. Query with ADQL (SQL for astronomy) through the TAP service. Table 'pscomppars' has one "
           "row per planet with the best parameters; 'ps' has every published solution.",
    provides=["6,000+ confirmed exoplanets: discovery year/method/facility, orbital period, radius, mass, equilibrium temperature",
              "Host-star properties (temperature, mass, distance), transit and radial-velocity data", "TESS and Kepler candidate tables"],
    ideas="Habitable-zone classification, discovery-method bias analysis, planet-radius distributions (the 'radius valley').",
    endpoint="GET https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
    params=[
        ("query", "ADQL query", "select pl_name, disc_year, pl_rade from pscomppars where pl_rade < 1.5"),
        ("format", "json | csv | tsv | votable", "json"),
        ("tables", "pscomppars (one row per planet) | ps (all solutions) | toi (TESS candidates) | ...", "pscomppars"),
        ("key columns", "pl_name, hostname, disc_year, discoverymethod, pl_orbper, pl_rade, pl_bmasse, pl_eqt, sy_dist", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def main():
    by_method = get(URL, params={"format": "json", "query":
        "select discoverymethod, count(*) as n from pscomppars group by discoverymethod order by n desc"})
    earthlike = get(URL, params={"format": "json", "query":
        "select top 5 pl_name, disc_year, pl_rade, pl_eqt, sy_dist from pscomppars "
        "where pl_rade between 0.8 and 1.25 and pl_eqt between 200 and 300 order by sy_dist"})
    show({"confirmed_planets": sum(r["n"] for r in by_method), "by_discovery_method": by_method[:5],
          "nearest_earth_size_temperate": earthlike})


if __name__ == "__main__":
    main()
