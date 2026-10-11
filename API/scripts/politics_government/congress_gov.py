META = dict(
    name="Congress.gov API",
    domain="Political Science & Government",
    docs="https://github.com/LibraryOfCongress/api.congress.gov",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Sign up at https://api.congress.gov/sign-up/ (an api.data.gov key — the same key also works for "
           "OpenFEC, Regulations.gov, NREL, NASA). 5,000 req/hour; DEMO_KEY: 30/hour.",
    provides=["Bills & resolutions: title, sponsor, cosponsors, actions, status, subjects, text versions",
              "Members of Congress (party, state, terms)", "Roll-call votes (House), committees, hearings, nominations, treaties"],
    ideas="Legislative effectiveness, bipartisanship networks from cosponsorship, policy-agenda topic modeling.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    k = os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")
    bills = get("https://api.congress.gov/v3/bill/119", params={"api_key": k, "limit": 5, "format": "json",
                                                              "sort": "updateDate+desc"})["bills"]
    show([{"bill": f'{b["type"]} {b["number"]}', "title": b["title"],
           "latestAction": b["latestAction"]["text"], "date": b["latestAction"]["actionDate"]} for b in bills])


if __name__ == "__main__":
    main()
