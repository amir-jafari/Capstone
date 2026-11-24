META = dict(
    name="Greenhouse Job Board API (company career pages)",
    domain="Jobs & Labor",
    docs="https://docs.greenhouse.io/job-board.html",
    auth="None",
    signup="No registration. Thousands of companies (Airbnb, Stripe, Pinterest, ...) host their careers page on Greenhouse; "
           "the board token is the company slug in their careers URL (boards.greenhouse.io/{token}).",
    provides=["Every open job at a company: title, location, department, office, last updated, full description (content=true)",
              "Departments and offices with their jobs", "Custom metadata fields some companies publish (e.g. remote eligibility)"],
    ideas="Tracking hiring by company/department over time, skill extraction from descriptions, hiring freezes as a signal.",
    endpoint="GET https://boards-api.greenhouse.io/v1/boards/{board_token}/{jobs | departments | offices}",
    params=[
        ("board_token", "Company slug", "airbnb, stripe, pinterest"),
        ("jobs?content=true", "Include the HTML job description", "content=true"),
        ("jobs/{id}?questions=true", "One job, with application questions", "—"),
        ("departments / offices", "Jobs grouped by department or office", "departments"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    jobs = get("https://boards-api.greenhouse.io/v1/boards/airbnb/jobs")["jobs"]
    depts = get("https://boards-api.greenhouse.io/v1/boards/airbnb/departments")["departments"]
    show({"company": "Airbnb", "open_jobs": len(jobs),
          "jobs_by_department": sorted(((d["name"], len(d["jobs"])) for d in depts if d["jobs"]), key=lambda x: -x[1])[:5],
          "top_locations": Counter(j["location"]["name"] for j in jobs).most_common(4),
          "latest": [{"title": j["title"], "location": j["location"]["name"], "updated": j["updated_at"][:10]}
                     for j in sorted(jobs, key=lambda j: j["updated_at"], reverse=True)[:3]]})


if __name__ == "__main__":
    main()
