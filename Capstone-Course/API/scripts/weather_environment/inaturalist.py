META = dict(
    name="iNaturalist API",
    domain="Weather & Climate",
    docs="https://api.inaturalist.org/v1/docs/",
    auth="None for reading (OAuth to post)",
    signup="No registration to read. Keep to ~1 request/second and ~10,000 requests/day. Research-grade observations "
           "also flow into GBIF.",
    provides=["Citizen-science observations with photos: species, location, date, observer, quality grade",
              "Species counts for any place, taxon search, identification agreements", "Place and project lookups"],
    ideas="Urban biodiversity, phenology (first-bloom dates over time), invasive-species spread, image datasets.",
    endpoint="GET https://api.inaturalist.org/v1/{resource}",
    params=[
        ("observations / observations/species_counts", "Observations, or species ranked by observation count", "observations/species_counts"),
        ("place_id / lat & lng & radius", "Place (find IDs via /v1/places/autocomplete?q=) or circle in km", "lat=38.9&lng=-77.04&radius=10"),
        ("taxon_id / taxon_name / iconic_taxa", "Taxon filters (iconic_taxa: Aves, Plantae, Insecta, ...)", "iconic_taxa=Aves"),
        ("quality_grade", "research | needs_id | casual", "research"),
        ("d1 / d2", "Observed-on date range YYYY-MM-DD", "2026-01-01"),
        ("per_page / page / order_by", "Max 200 per page; order_by observed_on | created_at | votes", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.inaturalist.org/v1/observations/species_counts", params={
        "lat": 38.9, "lng": -77.04, "radius": 10, "quality_grade": "research", "iconic_taxa": "Aves",
        "d1": "2026-01-01", "per_page": 5})
    show({"bird_species_observed_near_DC_2026": d["total_results"], "most_observed": [
        {"species": r["taxon"]["name"], "common_name": r["taxon"].get("preferred_common_name"), "observations": r["count"]}
        for r in d["results"]]})


if __name__ == "__main__":
    main()
