META = dict(
    name="Adzuna Jobs API",
    domain="Jobs & Labor",
    docs="https://developer.adzuna.com/docs/search",
    auth="app_id + app_key (query parameters)",
    env="ADZUNA_APP_ID, ADZUNA_APP_KEY",
    signup="Register at https://developer.adzuna.com/signup for an app ID and key. Free tier ~250 requests/day; attribution "
           "to Adzuna required.",
    provides=["Millions of live job ads in 19 countries aggregated from many boards: title, company, location, salary, contract type",
              "Salary histograms and averages by keyword/location, top companies, regional job counts, salary history"],
    ideas="Real-time labor-demand indices, salary prediction from job text, remote-work share by occupation.",
    endpoint="GET https://api.adzuna.com/v1/api/jobs/{country}/{search/{page} | histogram | top_companies | history | geodata}",
    params=[
        ("country", "us | gb | ca | de | fr | au | in | ...", "us"),
        ("what / what_exclude / title_only", "Keywords", "data scientist"),
        ("where / distance", "Location text, radius in km", "Washington DC / 25"),
        ("salary_min / full_time / permanent", "Filters", "100000"),
        ("sort_by", "relevance | date | salary", "date"),
        ("results_per_page", "Max 50", "5"),
        ("app_id / app_key", "Your credentials", "—"),
    ],
    sample="""{
  "count": 2147,
  "mean_salary": 141250.3,
  "jobs": [
    {"title": "Senior Data Scientist", "company": "Booz Allen Hamilton", "location": "Washington, DC",
     "salary_min": 120000, "salary_max": 170000, "created": "2026-10-10T14:22:31Z"}
  ]
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.adzuna.com/v1/api/jobs/us/search/1", params={
        "app_id": key("ADZUNA_APP_ID"), "app_key": key("ADZUNA_APP_KEY"), "what": "data scientist",
        "where": "Washington DC", "results_per_page": 5, "sort_by": "date"})
    show({"count": d["count"], "mean_salary": d.get("mean"), "jobs": [
        {"title": j["title"], "company": j["company"]["display_name"], "location": j["location"]["display_name"],
         "salary_min": j.get("salary_min"), "salary_max": j.get("salary_max"), "created": j["created"]} for j in d["results"]]})


if __name__ == "__main__":
    main()
