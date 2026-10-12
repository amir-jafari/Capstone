META = dict(
    name="Coursera Catalog API",
    domain="Education",
    docs="https://api.coursera.org/api/courses.v1 (the old docs site build.coursera.org no longer resolves)",
    auth="None",
    signup="No registration for the public catalog. Keyword search (q=search) is no longer accepted (HTTP 405), so page "
           "through the catalog and filter locally.",
    provides=["Every Coursera course: name, slug, description, workload, languages, domain/subdomain tags",
              "Partner (university/company) and instructor IDs, specializations"],
    ideas="Mapping the online-course supply by skill domain, comparing course topics with job-posting skills (see Jobs).",
    endpoint="GET https://api.coursera.org/api/courses.v1",
    params=[
        ("fields", "Extra fields: description, workload, primaryLanguages, domainTypes, partnerIds, instructorIds", "name,domainTypes,workload"),
        ("start / limit", "Offset and page size (paging.next gives the next start)", "0 / 100"),
        ("ids", "Look up specific course IDs", "—"),
        ("/api/partners.v1 · /api/instructors.v1", "Sibling endpoints: partner institutions and instructors", "fields=name,location"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.coursera.org/api/courses.v1", params={"fields": "name,domainTypes,workload", "limit": 100})
    domains = Counter(t["domainId"] for c in d["elements"] for t in c.get("domainTypes", []))
    show({"total_courses": d["paging"].get("total"), "domains_in_first_100": domains.most_common(6),
          "examples": [{"name": c["name"], "workload": c.get("workload"),
                        "subdomains": [t["subdomainId"] for t in c.get("domainTypes", [])]} for c in d["elements"][:3]]})


if __name__ == "__main__":
    main()
