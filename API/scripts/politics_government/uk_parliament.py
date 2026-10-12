META = dict(
    name="UK Parliament APIs",
    domain="Political Science & Government",
    docs="https://developer.parliament.uk/",
    auth="None",
    signup="No registration.",
    provides=["Members (MPs/Lords), party, constituency", "Commons & Lords divisions (votes) with every member's vote",
              "Written questions & answers, bills, committees"],
    ideas="Comparative legislative behavior, party-cohesion / rebellion analysis.",
    endpoint="GET https://commonsvotes-api.parliament.uk/data/divisions.json/search",
    params=[
        ("queryParameters.searchTerm", "Text in the division title", "Budget"),
        ("queryParameters.startDate / endDate", "Date range YYYY-MM-DD", "2026-01-01"),
        ("queryParameters.memberId", "Only divisions this MP voted in", "172"),
        ("queryParameters.take / skip", "Paging (max 25)", "5"),
        ("/data/division/{id}.json", "Sibling endpoint: every MP's vote in one division", "/data/division/1234.json"),
        ("members-api.parliament.uk/api/Members/Search?Name=", "Members API: find MPs/Lords, party, constituency", "Name=Starmer"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    divs = get("https://commonsvotes-api.parliament.uk/data/divisions.json/search", params={"queryParameters.take": 5})
    show([{"title": d["Title"], "date": d["Date"][:10], "ayes": d["AyeCount"], "noes": d["NoCount"]} for d in divs])


if __name__ == "__main__":
    main()
