META = dict(
    name="FBI Crime Data Explorer API",
    domain="Law & Justice",
    docs="https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/docApi",
    auth="api.data.gov key (`DEMO_KEY` works but is often rate-limited here)",
    env="DATA_GOV_API_KEY",
    signup="Use your api.data.gov key (https://api.data.gov/signup/), passed as API_KEY. The shared DEMO_KEY hits "
           "HTTP 429 quickly on this API, so get a personal key.",
    provides=["Monthly offense and clearance rates (per 100k) for the nation, states and individual agencies",
              "Reported counts by offense: violent, property, homicide, rape, robbery, aggravated assault, burglary, ...",
              "Agency directory (ORI codes), NIBRS incident detail, hate crime, use-of-force and LEOKA data"],
    ideas="Crime trends and policy evaluation, clearance-rate disparities, linking crime to housing/transport data.",
    endpoint="GET https://api.usa.gov/crime/fbi/cde/summarized/{level}/{offense}",
    params=[
        ("level", "state/{abbr}; the docs also list national and agency/{ORI}", "state/VA"),
        ("offense", "V violent crime, P property crime, RPE rape, ROB robbery, ASS aggravated assault (full code list in the docs)", "V"),
        ("from / to", "Month range MM-YYYY", "01-2023 / 12-2023"),
        ("API_KEY", "api.data.gov key (query parameter)", "DEMO_KEY"),
        ("/agency/byStateAbbr/{abbr}", "Sibling endpoint (per the docs): agencies and ORI codes in a state", "/agency/byStateAbbr/DC"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.usa.gov/crime/fbi/cde/summarized/state/VA/V", params={
        "from": "01-2023", "to": "12-2023", "API_KEY": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")})
    rates = d["offenses"]["rates"]
    show({"offense": "violent crime, monthly rate per 100k (2023)", "annual_rate_per_100k": {
        name: round(sum(months.values()), 1) for name, months in rates.items()}})


if __name__ == "__main__":
    main()
