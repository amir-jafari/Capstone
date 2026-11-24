META = dict(
    name="U.S. State Department Travel Advisories",
    domain="Transportation & Geospatial",
    docs="https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html",
    auth="None",
    signup="No registration. JSON feed behind the official advisories page; Category holds the country code(s). "
           "Keep a dated copy to study how levels change over time.",
    provides=["Advisory level 1-4 for every country (Exercise Normal Precautions ... Do Not Travel)",
              "Risk indicators in the summary text (crime, terrorism, unrest, health, kidnapping...), last-updated dates, links"],
    ideas="Travel-risk vs. tourism flows, how advisories respond to events (ACLED, GDELT), text classification of risks.",
    endpoint="GET https://cadataapi.state.gov/api/TravelAdvisories",
    params=[
        ("(no parameters)", "Returns all advisories", "—"),
        ("Title", "Country + level, e.g. 'Yemen - Level 4: Do Not Travel'", "—"),
        ("Category", "Two-letter country code(s)", "YM"),
        ("Summary / Published / Updated", "HTML summary text and dates", "—"),
    ],
)
import re
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    adv = get("https://cadataapi.state.gov/api/TravelAdvisories")
    level = lambda a: int(m.group(1)) if (m := re.search(r"Level (\d)", a["Title"])) else None
    levels = Counter(level(a) for a in adv)
    show({"advisories": len(adv), "by_level": {f"Level {k}": v for k, v in sorted(levels.items(), key=lambda kv: kv[0] or 0)},
          "level_4_do_not_travel": sorted({a["Title"].split(" - ")[0] for a in adv if level(a) == 4})},  # feed has duplicates
         max_items=25)


if __name__ == "__main__":
    main()
