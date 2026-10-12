META = dict(
    name="GBIF (Global Biodiversity Information Facility)",
    domain="Weather & Climate",
    docs="https://techdocs.gbif.org/en/openapi/",
    auth="None for search (account for bulk downloads)",
    signup="No registration to search (max 100,000 records via paging). For large extracts, create an account and use "
           "the asynchronous download API, which mints a citable DOI.",
    provides=["2.5B+ species occurrence records: species, coordinates, date, country, basis of record, dataset",
              "Species taxonomy and name matching", "Faceted counts (by year, country, species) without downloading records"],
    ideas="Species distribution modelling, climate-driven range shifts, biodiversity trends by region.",
    endpoint="GET https://api.gbif.org/v1/occurrence/search",
    params=[
        ("taxonKey / scientificName", "Species or higher taxon (find keys via /v1/species/match?name=)", "scientificName=Haliaeetus leucocephalus"),
        ("country / stateProvince / geometry", "ISO country, province name, or WKT polygon", "US"),
        ("year / eventDate", "Year or range", "2015,2025"),
        ("hasCoordinate / basisOfRecord", "true; HUMAN_OBSERVATION | PRESERVED_SPECIMEN | ...", "true"),
        ("facet + facetLimit", "Counts by a field (set limit=0 for counts only)", "facet=year"),
        ("limit / offset", "Max 300 per page", "0"),
        ("/v1/species/match?name=", "Sibling endpoint: resolve a name to a taxonKey", "name=bald eagle"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.gbif.org/v1/occurrence/search", params={
        "scientificName": "Haliaeetus leucocephalus", "country": "US", "year": "2016,2025",
        "hasCoordinate": "true", "limit": 0, "facet": "year", "facetLimit": 10})
    # The latest year looks small until large datasets (e.g. eBird) publish their annual updates
    years = sorted(d["facets"][0]["counts"], key=lambda c: c["name"])
    show({"species": "Bald eagle (Haliaeetus leucocephalus), US", "occurrences_2016_2025": d["count"],
          "by_year": {c["name"]: c["count"] for c in years}}, max_items=10)


if __name__ == "__main__":
    main()
