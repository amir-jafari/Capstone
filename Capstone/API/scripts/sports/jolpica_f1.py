META = dict(
    name="Jolpica-F1 (Ergast-compatible F1 history)",
    domain="Sports",
    docs="https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md",
    auth="None",
    signup="No registration. Drop-in replacement for the retired Ergast API; ~4 req/sec, 500/hour.",
    provides=["Every F1 season since 1950: race results, qualifying, sprint, driver & constructor standings",
              "Lap times and pit stops (1996+/2011+), circuits, drivers, constructors"],
    ideas="Long-run analysis of team dominance, driver-rating models, era comparisons.",
    endpoint="GET https://api.jolpi.ca/ergast/f1/{season}/{round}/{resource}.json",
    params=[
        ("season", "Year or current", "2025"),
        ("round", "Race number or last (optional)", "last"),
        ("resource", "results | qualifying | sprint | driverStandings | constructorStandings | laps | pitstops | drivers | circuits", "driverStandings"),
        ("drivers/{id} / constructors/{id} / circuits/{id}", "Filter path segments (combinable)", "drivers/max_verstappen/results"),
        ("limit / offset", "Paging (max 100)", "30"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.jolpi.ca/ergast/f1/2025/driverStandings.json")["MRData"]["StandingsTable"]["StandingsLists"][0]
    show({"season": d["season"], "after_round": d["round"], "standings": [
        {"pos": s["position"], "driver": f'{s["Driver"]["givenName"]} {s["Driver"]["familyName"]}',
         "team": s["Constructors"][0]["name"], "points": s["points"], "wins": s["wins"]}
        for s in d["DriverStandings"]]})


if __name__ == "__main__":
    main()
