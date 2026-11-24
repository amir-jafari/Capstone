META = dict(
    name="Regulations.gov API v4",
    domain="Political Science & Government",
    docs="https://open.gsa.gov/api/regulationsgov/",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use an api.data.gov key (https://api.data.gov/signup/). 1,000 req/hour.",
    provides=["Federal rulemaking dockets & documents", "Public comments submitted on proposed rules (text + attachments)",
              "Agency, docket ID, comment period dates"],
    ideas="Public-comment analysis, detecting mass/astroturf comment campaigns, interest-group influence.",
    endpoint="GET https://api.regulations.gov/v4/documents  ·  /v4/comments  ·  /v4/dockets",
    params=[
        ("filter[searchTerm]", "Keyword search", "artificial intelligence"),
        ("filter[documentType]", "Notice | Rule | Proposed Rule | Supporting & Related Material | Other", "Proposed Rule"),
        ("filter[agencyId]", "Agency acronym", "EPA"),
        ("filter[postedDate][ge] / [le]", "Date range YYYY-MM-DD", "2026-01-01"),
        ("filter[docketId]", "Restrict to one docket", "EPA-HQ-OAR-2021-0317"),
        ("sort", "postedDate | -postedDate | title | ...", "-postedDate"),
        ("page[size] / page[number]", "5-250 per page; max 20 pages (narrow by date to go further)", "5"),
        ("/v4/comments?filter[commentOnId]=", "Sibling endpoint: comments on a document (objectId from /documents)", "filter[commentOnId]=0900006…"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    d = get("https://api.regulations.gov/v4/documents", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "filter[documentType]": "Proposed Rule",
        "sort": "-postedDate", "page[size]": 5})
    show([{k: x["attributes"][k] for k in ("title", "agencyId", "postedDate", "commentEndDate")} | {"id": x["id"]}
          for x in d["data"]])


if __name__ == "__main__":
    main()
