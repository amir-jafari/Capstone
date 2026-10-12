META = dict(
    name="The Muse Public API (jobs)",
    domain="Jobs & Labor",
    docs="https://www.themuse.com/developers/api/v2",
    auth="None (optional key raises the limit)",
    env="THEMUSE_API_KEY (optional)",
    signup="Works without a key (500 requests/hour). Register at https://www.themuse.com/developers/api/v2/apps for a key "
           "(3,600/hour).",
    provides=["Job postings with category, experience level (entry/mid/senior/internship), locations, company, publication date",
              "Company profiles with size, industries, locations", "Full HTML descriptions"],
    ideas="Entry-level vs. senior demand by field, internship availability, company-culture text analysis.",
    endpoint="GET https://www.themuse.com/api/public/{jobs | companies}",
    params=[
        ("page", "Page number (required, 20 results per page)", "1"),
        ("category", "Job category (repeatable): Data and Analytics, Software Engineering, ...", "Data and Analytics"),
        ("level", "Entry Level | Mid Level | Senior Level | Management | Internship (repeatable)", "Entry Level"),
        ("location", "City, State (repeatable)", "Washington, DC"),
        ("company", "Company name (repeatable)", "—"),
        ("descending", "true -> newest first", "true"),
        ("api_key", "Optional key", "<THEMUSE_API_KEY>"),
    ],
)
import os
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    params = [("page", 1), ("category", "Data and Analytics"), ("level", "Entry Level"), ("descending", "true")]
    if os.getenv("THEMUSE_API_KEY"):
        params.append(("api_key", os.environ["THEMUSE_API_KEY"]))
    d = get("https://www.themuse.com/api/public/jobs", params=params)
    show({"entry_level_data_jobs": d["total"],
          "top_locations_on_page": Counter(l["name"] for j in d["results"] for l in j["locations"]).most_common(4),
          "examples": [{"title": j["name"], "company": j["company"]["name"],
                        "locations": [l["name"] for l in j["locations"]][:2], "published": j["publication_date"][:10]}
                       for j in d["results"][:3]]})


if __name__ == "__main__":
    main()
