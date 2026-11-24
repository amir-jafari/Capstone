META = dict(
    name="GeoNames (place names gazetteer)",
    domain="Transportation & Geospatial",
    docs="https://www.geonames.org/export/web-services.html",
    auth="Free username (query `username`)",
    env="GEONAMES_USERNAME",
    signup="Create an account at https://www.geonames.org/login, then enable free web services on your account page. "
           "Limits: 10,000 credits/day, 1,000/hour. The full database is also a free download (CC BY 4.0).",
    provides=["12M+ place names: cities, landmarks, mountains, airports, with population, coordinates, admin regions, feature codes",
              "Reverse geocoding (nearby places), time zones, elevation, postal codes, country info",
              "Alternate names in many languages"],
    ideas="City-level datasets with population, building place hierarchies, time-zone and elevation enrichment.",
    endpoint="GET http://api.geonames.org/{searchJSON | findNearbyPlaceNameJSON | timezoneJSON | countryInfoJSON}",
    params=[
        ("q / name_equals", "Search text", "springfield"),
        ("country / featureClass / featureCode", "Filters: P populated places, H water, T mountains; PPLA capital of first-order division", "country=US&featureClass=P"),
        ("orderby", "population | elevation | relevance", "population"),
        ("lat / lng", "For nearby / timezone / elevation services", "38.9 / -77.04"),
        ("maxRows / startRow", "Paging (max 1,000)", "5"),
        ("username", "Your account (the shared 'demo' account is for testing only)", "<GEONAMES_USERNAME>"),
    ],
    sample="""[
  {"name": "Springfield", "adminName1": "Illinois", "population": 114394, "lat": "39.80172", "lng": "-89.64371"},
  {"name": "Springfield", "adminName1": "Missouri", "population": 169176, "lat": "37.21533", "lng": "-93.29824"}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("http://api.geonames.org/searchJSON", params={
        "name_equals": "Springfield", "country": "US", "featureClass": "P", "orderby": "population", "maxRows": 5,
        "username": key("GEONAMES_USERNAME")})
    show([{k: g.get(k) for k in ("name", "adminName1", "population", "lat", "lng")} for g in d["geonames"]])


if __name__ == "__main__":
    main()
