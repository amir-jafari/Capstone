META = dict(
    name="Kaggle API",
    domain="AI & Machine Learning",
    docs="https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md",
    auth="Basic auth with username + API key (dataset search currently answers without auth)",
    env="KAGGLE_USERNAME, KAGGLE_KEY",
    signup="Kaggle account -> Settings -> API -> 'Create New Token' downloads kaggle.json (username + key). "
           "CLI: `pip install kaggle`, then `kaggle datasets download -d owner/dataset`. Accept each competition's rules on the website first.",
    provides=["Search 400k+ public datasets: title, owner, size, licence, usability rating, downloads, votes",
              "Download datasets and competition files", "Competitions, leaderboards, notebooks (kernels), models"],
    ideas="Finding training data fast; studying which dataset properties predict popularity.",
    endpoint="GET https://www.kaggle.com/api/v1/datasets/list",
    params=[
        ("search", "Search term", "titanic"),
        ("sortBy", "hottest | votes | updated | active | published", "votes"),
        ("filetype", "all | csv | sqlite | json | bigQuery | parquet", "csv"),
        ("license / tagids / user", "Filters: licence group, tag IDs, owner", "license=cc"),
        ("page", "Page number (20 per page)", "1"),
        ("/api/v1/datasets/download/{owner}/{dataset}", "Download a dataset as a zip (auth required)", "/datasets/download/heptapod/titanic"),
        ("/api/v1/competitions/list?search=", "Sibling endpoint: competitions (auth required)", "search=forecasting"),
    ],
)
import os

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show


def main():
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]) if os.getenv("KAGGLE_KEY") else None
    r = requests.get("https://www.kaggle.com/api/v1/datasets/list", auth=auth, timeout=30, headers={"User-Agent": USER_AGENT},
                     params={"search": "climate", "sortBy": "votes", "filetype": "csv"})
    r.raise_for_status()
    # Most field names in this response carry a "Nullable" suffix (ref does not)
    show([{"ref": d.get("ref"), "title": d.get("titleNullable"),
           "size_bytes": d.get("totalBytesNullable"), "license": d.get("licenseNameNullable"),
           "usability": round(d.get("usabilityRatingNullable") or 0, 2), "downloads": d.get("downloadCount"),
           "votes": d.get("voteCount")} for d in r.json()[:5]])


if __name__ == "__main__":
    main()
