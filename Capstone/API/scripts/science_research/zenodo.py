META = dict(
    name="Zenodo (research data & software repository)",
    domain="Science & Research",
    docs="https://developers.zenodo.org/",
    auth="None for search/download (token to deposit)",
    signup="No registration to search and download public records. CERN-run; every record has a DOI. Also great for "
           "publishing your own capstone dataset or code (GitHub integration).",
    provides=["3M+ records: datasets, software, papers, posters, presentations, each with DOI, licence, versions",
              "File lists with direct download links and checksums", "Download/view statistics, communities, related identifiers"],
    ideas="Finding reusable datasets, studying research-data reuse, citing/archiving your own capstone outputs.",
    endpoint="GET https://zenodo.org/api/records",
    params=[
        ("q", "Elasticsearch query string; fields like resource_type.type:dataset, creators.name:", "climate AND resource_type.type:dataset"),
        ("type", "dataset | software | publication | poster | ...", "dataset"),
        ("sort", "bestmatch | mostrecent | -mostrecent", "mostrecent"),
        ("communities", "Restrict to a community", "—"),
        ("size / page", "Page size (max 25 anonymous)", "5"),
        ("/api/records/{id}", "One record with files and stats", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://zenodo.org/api/records", params={"q": "\"air quality\"", "type": "dataset", "sort": "mostrecent", "size": 5})
    show({"matching_datasets": d["hits"]["total"], "latest": [
        {"title": r["metadata"]["title"], "doi": r.get("doi"), "published": r["metadata"].get("publication_date"),
         "files": len(r.get("files", [])), "license": (r["metadata"].get("license") or {}).get("id"),
         "downloads": r.get("stats", {}).get("downloads")} for r in d["hits"]["hits"]]})


if __name__ == "__main__":
    main()
