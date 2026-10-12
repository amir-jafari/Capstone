META = dict(
    name="USAJOBS Search API (federal jobs)",
    domain="Political Science & Government",
    docs="https://developer.usajobs.gov/",
    auth="API key (header `Authorization-Key`) + your email as `User-Agent`",
    env="USAJOBS_API_KEY, USAJOBS_EMAIL",
    signup="Request a key at https://developer.usajobs.gov/apirequest/ (emailed). Send headers Host: data.usajobs.gov, "
           "User-Agent: <the email you registered>, Authorization-Key: <key>.",
    provides=["Every open federal job posting: title, agency, grade/pay range, location, remote/telework, security clearance",
              "Qualification summaries, hiring paths (public, veterans, students), open/close dates",
              "Historic job announcements (separate endpoint)"],
    ideas="Federal hiring trends by agency and skill (e.g. demand for data scientists), pay comparisons by location.",
    endpoint="GET https://data.usajobs.gov/api/search",
    params=[
        ("Keyword / PositionTitle", "Search text / title", "data scientist"),
        ("LocationName / Radius", "City or state, radius in miles", "Washington, DC / 25"),
        ("Organization", "Agency subelement codes (semicolon-separated)", "HE38"),
        ("RemunerationMinimum / PayGradeLow / PayGradeHigh", "Pay filters", "100000"),
        ("HiringPath", "public | vet | student | fed | ...", "public"),
        ("DatePosted / SortField", "Days since posted; sort field (DatePosted, Salary, ...)", "7 / DatePosted"),
        ("ResultsPerPage / Page", "Max 500 per page", "5"),
    ],
    sample="""{
  "total": 212,
  "jobs": [
    {"title": "Data Scientist", "agency": "Census Bureau", "location": "Suitland, Maryland",
     "pay": "$104,604 - $163,282 Per Year", "close_date": "2026-10-24"}
  ]
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://data.usajobs.gov/api/search", headers={
        "Host": "data.usajobs.gov", "User-Agent": key("USAJOBS_EMAIL"), "Authorization-Key": key("USAJOBS_API_KEY")},
        params={"Keyword": "data scientist", "LocationName": "Washington, DC", "ResultsPerPage": 5})["SearchResult"]
    show({"total": d["SearchResultCountAll"], "jobs": [
        {"title": j["PositionTitle"], "agency": j["OrganizationName"], "location": j["PositionLocationDisplay"],
         "pay": f'${j["PositionRemuneration"][0]["MinimumRange"]} - ${j["PositionRemuneration"][0]["MaximumRange"]}',
         "close_date": j["ApplicationCloseDate"][:10]}
        for j in (item["MatchedObjectDescriptor"] for item in d["SearchResultItems"])]})


if __name__ == "__main__":
    main()
