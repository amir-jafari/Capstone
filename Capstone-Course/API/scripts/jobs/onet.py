META = dict(
    name="O*NET Web Services (occupations & skills)",
    domain="Jobs & Labor",
    docs="https://services.onetcenter.org/reference/",
    auth="API key (header `X-API-Key`)",
    env="ONET_API_KEY",
    signup="Register at https://services.onetcenter.org/developer/signup (free; attribution required). The current API "
           "takes an X-API-Key header on GET requests; the older v1.9 Basic-auth API is listed under previous versions.",
    provides=["~1,000 US occupations (O*NET-SOC codes): tasks, skills, knowledge, abilities, work activities, technology used",
              "Keyword search from job titles to occupations; related occupations; education and training requirements",
              "Bright Outlook flags, wages and employment outlook links (My Next Move / CareerOneStop)"],
    ideas="Skill-based job similarity, automation-exposure scores, matching job ads (Jobs section) to standard occupations.",
    endpoint="GET https://api-v2.onetcenter.org/{resource}",
    params=[
        ("online/search?keyword=", "Find occupations from a title or code", "keyword=data scientist"),
        ("online/occupations/{code}/summary/skills", "Skills for an occupation (also knowledge, abilities, technology_skills, tasks)", "15-2051.00/summary/skills"),
        ("online/occupations/{code}/details/...", "Full ratings (importance/level) instead of summaries", "—"),
        ("start / end", "Result range for paged lists", "1 / 20"),
        ("X-API-Key (header)", "Your key (query-string keys are rejected)", "<ONET_API_KEY>"),
    ],
    sample="""{
  "keyword": "data scientist",
  "matches": [
    {"code": "15-2051.00", "title": "Data Scientists"},
    {"code": "15-2041.00", "title": "Statisticians"},
    {"code": "15-1221.00", "title": "Computer and Information Research Scientists"}
  ]
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api-v2.onetcenter.org/online/search", headers={"X-API-Key": key("ONET_API_KEY")},
            params={"keyword": "data scientist"})
    show({"keyword": "data scientist", "matches": [{"code": o["code"], "title": o["title"]} for o in d["occupation"][:5]]})


if __name__ == "__main__":
    main()
