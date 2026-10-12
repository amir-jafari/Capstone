META = dict(
    name="Harvard Dataverse (MIT Election Lab & replication data)",
    domain="Political Science & Government",
    docs="https://guides.dataverse.org/en/latest/api/",
    auth="None for search and public downloads (token to deposit)",
    env="DATAVERSE_API_TOKEN (optional)",
    signup="No registration to search or download public files. Hosts the MIT Election Data + Science Lab returns "
           "(president, Senate, House, county and precinct level) and replication data for thousands of political-science papers.",
    provides=["Search datasets by keyword, subject, author, publication; citation and DOI for each",
              "List and download each dataset's files (CSV, Stata, R) via the Data Access API",
              "Election returns since 1976 from the MIT Election Lab"],
    ideas="Election forecasting, turnout and polarization analysis, replicating published political-science findings.",
    endpoint="GET https://dataverse.harvard.edu/api/search",
    params=[
        ("q", "Search text", "MIT Election Data"),
        ("type", "dataset | file | dataverse", "dataset"),
        ("subtree", "Restrict to one collection alias", "medsl"),
        ("sort / order", "name | date; asc | desc", "date / desc"),
        ("per_page / start", "Paging (max 1,000)", "5"),
        ("/api/datasets/:persistentId/?persistentId=doi:...", "Sibling endpoint: dataset metadata and file list", "doi:10.7910/DVN/42MVDX"),
        ("/api/access/datafile/{fileId}", "Download a file (format=original for the uploaded format)", "format=original"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://dataverse.harvard.edu/api/search", params={
        "q": "election returns", "type": "dataset", "subtree": "medsl", "sort": "date", "order": "desc", "per_page": 5})["data"]
    show({"datasets": d["total_count"], "latest": [
        {"name": x["name"], "published": x.get("published_at", "")[:10], "doi": x.get("global_id"),
         "files": x.get("fileCount")} for x in d["items"]]})


if __name__ == "__main__":
    main()
