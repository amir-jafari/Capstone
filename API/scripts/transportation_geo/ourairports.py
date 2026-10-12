META = dict(
    name="OurAirports (global airports dataset)",
    domain="Transportation & Geospatial",
    docs="https://github.com/davidmegginson/ourairports-data (data dictionary: ourairports.com/help/data-dictionary.html)",
    auth="None (CSV downloads)",
    signup="No registration; public-domain CSVs updated nightly (mirrored on GitHub Pages). Companion files: runways, "
           "frequencies, navaids, countries, regions.",
    provides=["~80,000 airports, heliports and seaplane bases: ICAO/IATA codes, type (large/medium/small), coordinates, elevation",
              "Municipality, country, region, scheduled-service flag, Wikipedia links", "Runway lengths and surfaces"],
    ideas="Airport accessibility analysis, joining codes across flight datasets, mapping air-service deserts.",
    endpoint="GET https://davidmegginson.github.io/ourairports-data/{file}.csv",
    params=[
        ("airports.csv", "Columns: ident, type, name, latitude_deg, longitude_deg, iso_country, municipality, scheduled_service, iata_code ...", "—"),
        ("runways.csv", "airport_ident, length_ft, surface, lighted, ...", "—"),
        ("countries.csv / regions.csv", "Code lookups", "—"),
        ("type values", "large_airport | medium_airport | small_airport | heliport | seaplane_base | closed", "large_airport"),
    ],
)
import csv
import io
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    rows = list(csv.DictReader(io.StringIO(get("https://davidmegginson.github.io/ourairports-data/airports.csv",
                                               as_json=False))))
    large = [r for r in rows if r["type"] == "large_airport"]
    show({"airports_total": len(rows), "by_type": Counter(r["type"] for r in rows).most_common(),
          "large_airports_by_country": Counter(r["iso_country"] for r in large).most_common(5),
          "dc_area": [{k: r[k] for k in ("ident", "iata_code", "name", "type")} for r in rows
                      if r["iata_code"] in ("DCA", "IAD", "BWI")]})


if __name__ == "__main__":
    main()
