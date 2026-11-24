META = dict(
    name="Census LEHD LODES (jobs by census block)",
    domain="Jobs & Labor",
    docs="https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.3.pdf",
    auth="None (CSV downloads)",
    signup="No registration; no REST API - gzipped CSVs per state, year and file type. Block GEOIDs join directly to Census "
           "tracts (first 11 digits) and the Census Geocoder output.",
    provides=["WAC: jobs by workplace census block, split by age, earnings band, industry (NAICS sector), race, education",
              "RAC: the same counts by where workers live", "OD: origin-destination commuting flows between home and work blocks"],
    ideas="Job accessibility (with OSRM travel times), commuting networks, spatial mismatch between homes and jobs.",
    endpoint="GET https://lehd.ces.census.gov/data/lodes/LODES8/{state}/{wac | rac | od}/{file}.csv.gz",
    params=[
        ("state", "Two-letter state code, lower case", "dc"),
        ("file name", "{state}_{wac|rac}_{segment}_{jobtype}_{year}.csv.gz", "dc_wac_S000_JT00_2022.csv.gz"),
        ("segment", "S000 all workers, SA01-03 age bands, SE01-03 earnings bands, SI01-03 industry groups", "S000"),
        ("job type", "JT00 all jobs, JT01 primary jobs, JT02 private, ...", "JT00"),
        ("columns", "w_geocode, C000 total jobs, CNS01-CNS20 jobs by NAICS sector, CE01-03 earnings bands, ...", "—"),
        ("OD files", "{state}_od_{main|aux}_{jobtype}_{year}.csv.gz: w_geocode, h_geocode, S000", "dc_od_main_JT00_2022.csv.gz"),
    ],
)
import csv
import gzip
import io

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

URL = "https://lehd.ces.census.gov/data/lodes/LODES8/dc/wac/dc_wac_S000_JT00_2022.csv.gz"
SECTORS = {"CNS12": "Professional/scientific/technical", "CNS20": "Public administration", "CNS16": "Health care",
           "CNS15": "Educational services", "CNS18": "Accommodation & food"}


def main():
    raw = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=120).content
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))
    total = sum(int(r["C000"]) for r in rows)
    tracts = {}
    for r in rows:  # block GEOID -> tract GEOID = first 11 digits
        tracts[r["w_geocode"][:11]] = tracts.get(r["w_geocode"][:11], 0) + int(r["C000"])
    show({"DC_jobs_2022": total, "blocks_with_jobs": len(rows),
          "jobs_by_sector": {name: sum(int(r[c]) for r in rows) for c, name in SECTORS.items()},
          "top_tracts_by_jobs": sorted(tracts.items(), key=lambda kv: -kv[1])[:3]})


if __name__ == "__main__":
    main()
