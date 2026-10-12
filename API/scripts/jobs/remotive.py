META = dict(
    name="Remotive (remote jobs)",
    domain="Jobs & Labor",
    docs="https://github.com/remotive-com/remote-jobs-api",
    auth="None",
    signup="No registration. Max ~4 requests/day recommended (jobs update daily). Terms: link back to Remotive and do not "
           "republish jobs on other job boards.",
    provides=["Remote job listings: title, company, category, job type, candidate location restrictions, salary text, tags",
              "Full HTML descriptions, publication date"],
    ideas="Remote-work demand by category, location restrictions in 'remote' jobs, salary disclosure rates.",
    endpoint="GET https://remotive.com/api/remote-jobs",
    params=[
        ("category", "Category slug (list via /api/remote-jobs/categories): software-dev, data, design, marketing, ...", "data"),
        ("search", "Keyword in title/description", "python"),
        ("company_name", "Filter by company", "—"),
        ("limit", "Max jobs returned", "50"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://remotive.com/api/remote-jobs", params={"category": "data", "limit": 100})
    jobs = d["jobs"]
    show({"remote_data_jobs": d.get("job-count", len(jobs)),
          "candidate_location": Counter(j["candidate_required_location"] for j in jobs).most_common(4),
          "with_salary_pct": round(100 * sum(bool(j["salary"]) for j in jobs) / max(len(jobs), 1)),
          "examples": [{k: j[k] for k in ("title", "company_name", "job_type", "salary", "publication_date")} for j in jobs[:3]]})


if __name__ == "__main__":
    main()
