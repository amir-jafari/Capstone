META = dict(
    name="Launch Library 2 (The Space Devs)",
    domain="Science & Research",
    docs="https://ll.thespacedevs.com/docs/",
    auth="None (15 requests/hour) / paid tiers",
    signup="Works anonymously at 15 requests/hour. Use lldev.thespacedevs.com for development (stale data, no limit). "
           "Higher limits via Patreon-supported keys.",
    provides=["Every orbital launch since 1957 and upcoming launches: rocket, provider, pad, mission, orbit, status, time window",
              "Agencies, rockets (configurations), astronauts, spacecraft, space stations, docking events, landings (booster reuse)"],
    ideas="Launch cadence by provider, success-rate modeling, reusability economics, launch-delay prediction.",
    endpoint="GET https://ll.thespacedevs.com/2.3.0/{resource}/",
    params=[
        ("resource", "launches/upcoming | launches/previous | astronauts | agencies | spacecraft | ...", "launches/previous"),
        ("mode", "list | normal | detailed (amount of nested detail)", "list"),
        ("net__gte / net__lte", "Launch time filters (ISO 8601)", "2026-01-01T00:00:00Z"),
        ("lsp__name / location__ids / status__ids", "Provider, location, status filters", "lsp__name=SpaceX"),
        ("search", "Text search", "Starlink"),
        ("limit / offset", "Paging (max 100)", "10"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ll.thespacedevs.com/2.3.0/launches/previous/", params={
        "mode": "list", "net__gte": "2026-01-01T00:00:00Z", "limit": 100})
    launches = d["results"]
    providers = Counter(l["name"].split(" | ")[0] for l in launches)
    show({"launches_2026_so_far": d["count"], "rockets_in_latest_100": providers.most_common(5),
          "status_in_latest_100": Counter(l["status"]["abbrev"] for l in launches).most_common(),
          "latest": [{"name": l["name"], "net": l["net"], "status": l["status"]["abbrev"]} for l in launches[:3]]})


if __name__ == "__main__":
    main()
