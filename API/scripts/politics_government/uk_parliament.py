META = dict(
    name="UK Parliament APIs",
    domain="Political Science & Government",
    docs="https://developer.parliament.uk/",
    auth="None",
    signup="No registration.",
    provides=["Members (MPs/Lords), party, constituency", "Commons & Lords divisions (votes) with every member's vote",
              "Written questions & answers, bills, committees"],
    ideas="Comparative legislative behavior, party-cohesion / rebellion analysis.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    divs = get("https://commonsvotes-api.parliament.uk/data/divisions.json/search", params={"queryParameters.take": 5})
    show([{"title": d["Title"], "date": d["Date"][:10], "ayes": d["AyeCount"], "noes": d["NoCount"]} for d in divs])


if __name__ == "__main__":
    main()
