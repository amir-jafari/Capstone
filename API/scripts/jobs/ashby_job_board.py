META = dict(
    name="Ashby Job Postings API (startup career pages)",
    domain="Jobs & Labor",
    docs="https://developers.ashbyhq.com/docs/public-job-posting-api",
    auth="None",
    signup="No registration. Many startups (Ramp, Notion, Linear, ...) host careers pages on Ashby at jobs.ashbyhq.com/{org}. "
           "Compensation data is unusually complete because many postings include salary ranges.",
    provides=["Open jobs with department, team, location(s), employment type, remote flag, publish date",
              "Compensation summaries (salary ranges, equity) when includeCompensation=true", "Full HTML/plain descriptions"],
    ideas="Startup salary benchmarking, remote vs. on-site pay gaps, equity compensation patterns.",
    endpoint="GET https://api.ashbyhq.com/posting-api/job-board/{organization}",
    params=[
        ("organization", "Job-board name from jobs.ashbyhq.com/{organization}", "ramp"),
        ("includeCompensation", "true -> add compensation tiers and summary", "true"),
        ("response fields", "title, department, team, location, isRemote, employmentType, publishedAt, compensation", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    jobs = get("https://api.ashbyhq.com/posting-api/job-board/ramp", params={"includeCompensation": "true"})["jobs"]
    show({"company": "Ramp", "open_jobs": len(jobs),
          "by_department": Counter(j.get("department") for j in jobs).most_common(5),
          "remote_share": round(sum(bool(j.get("isRemote")) for j in jobs) / len(jobs), 2),
          "with_salary": [{"title": j["title"].strip(), "location": j.get("location"),
                           "compensation": (j.get("compensation") or {}).get("compensationTierSummary")}
                          for j in jobs if (j.get("compensation") or {}).get("compensationTierSummary")][:4]})


if __name__ == "__main__":
    main()
