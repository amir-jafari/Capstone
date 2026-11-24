META = dict(
    name="US Census Geocoder",
    domain="Transportation & Geospatial",
    docs="https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html",
    auth="None",
    signup="No registration. Batch mode geocodes up to 10,000 addresses per uploaded CSV.",
    provides=["US address -> matched address, lat/lon", "Census geographies for the point: state, county, tract, block group, block (GEOIDs)",
              "Congressional and state legislative districts"],
    ideas="The bridge between addresses and Census data: geocode once, then join ACS demographics by tract GEOID.",
    endpoint="GET https://geocoding.geo.census.gov/geocoder/{returntype}/{searchtype}",
    params=[
        ("returntype", "locations (coordinates only) | geographies (coordinates + census areas)", "geographies"),
        ("searchtype", "onelineaddress | address | coordinates | addressbatch (POST CSV)", "onelineaddress"),
        ("address / street, city, state, zip", "Address input", "2121 I St NW, Washington, DC 20052"),
        ("x / y", "Longitude / latitude for searchtype=coordinates", "-77.0486 / 38.8997"),
        ("benchmark / vintage", "Address and boundary versions", "Public_AR_Current / Current_Current"),
        ("layers", "Which geographies to return", "Census Tracts,Counties"),
        ("format", "json | html", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress", params={
        "address": "2121 I St NW, Washington, DC 20052", "benchmark": "Public_AR_Current",
        "vintage": "Current_Current", "format": "json"})["result"]["addressMatches"][0]
    geo = d["geographies"]
    block = next(v[0] for k, v in geo.items() if k.endswith("Census Blocks"))  # layer names carry the vintage year
    district = next(v[0] for k, v in geo.items() if k.endswith("Congressional Districts"))
    show({"matched": d["matchedAddress"], "lon": d["coordinates"]["x"], "lat": d["coordinates"]["y"],
          "county": geo["Counties"][0]["NAME"], "tract_GEOID": geo["Census Tracts"][0]["GEOID"],
          "block_group_GEOID": block["GEOID"][:12],  # a block GEOID's first 12 digits are its block group
          "congressional_district": district["NAME"]})


if __name__ == "__main__":
    main()
