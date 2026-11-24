META = dict(
    name="CareerOneStop Web API (U.S. Department of Labor)",
    domain="Jobs & Labor",
    docs="https://www.careeronestop.org/Developers/WebAPI/web-api.aspx",
    auth="User ID in the path + Bearer token",
    env="CAREERONESTOP_USER_ID, CAREERONESTOP_TOKEN",
    signup="Register at https://www.careeronestop.org/Developers/WebAPI/registration.aspx; you receive a user ID and a "
           "token by email. Free.",
    provides=["Occupation profiles with local wages (percentiles by state/metro), employment outlook, education required",
              "Training programs, certifications and licenses by occupation and location",
              "American Job Center locations, apprenticeship sponsors, skills-matcher"],
    ideas="Local wage comparisons, training ROI by occupation, workforce-development program mapping.",
    endpoint="GET https://api.careeronestop.org/v1/{service}/{userId}/...",
    params=[
        ("occupation/{userId}/{keyword}/{dataLevelOnly}/{start}/{limit}", "Search occupations by keyword (dataLevelOnly N/Y)", "data scientist/N/0/5"),
        ("occupation/{userId}/{onetCode}/{location}?wages=true&projectedEmployment=true", "Occupation details for a location", "15-2051.00/DC"),
        ("training/{userId}/{keyword}/{location}/{radius}/...", "Training programs near a location", "—"),
        ("Authorization (header)", "Bearer token", "Bearer <CAREERONESTOP_TOKEN>"),
    ],
    sample="""[
  {"OnetCode": "15-2051.00", "OnetTitle": "Data Scientists", "BrightOutlook": "Bright"},
  {"OnetCode": "15-2041.00", "OnetTitle": "Statisticians", "BrightOutlook": "Bright"}
]""",
)
from urllib.parse import quote

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    user, token = key("CAREERONESTOP_USER_ID"), key("CAREERONESTOP_TOKEN")
    d = get(f"https://api.careeronestop.org/v1/occupation/{user}/{quote('data scientist')}/N/0/5",
            headers={"Authorization": f"Bearer {token}"})
    show([{k: o.get(k) for k in ("OnetCode", "OnetTitle", "BrightOutlook")} for o in d["OccupationList"]])


if __name__ == "__main__":
    main()
