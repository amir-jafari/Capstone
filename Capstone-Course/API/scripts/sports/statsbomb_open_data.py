META = dict(
    name="StatsBomb Open Data (soccer event data)",
    domain="Sports",
    docs="https://github.com/statsbomb/open-data",
    auth="None (JSON files on GitHub)",
    signup="No registration; free for research and education with attribution to StatsBomb (see the repo licence). "
           "Python package: `pip install statsbombpy`.",
    provides=["Every on-ball event (pass, shot, carry, pressure...) with x/y location, time, players, outcome",
              "Shot xG, freeze frames, 360 data for some competitions", "Lineups; competitions incl. World Cups, Euros, Messi's La Liga career, women's leagues"],
    ideas="Expected-goals (xG) models, passing networks, pressing intensity, player-style clustering.",
    endpoint="GET https://raw.githubusercontent.com/statsbomb/open-data/master/data/{file}",
    params=[
        ("competitions.json", "All free competitions and seasons (competition_id, season_id)", "—"),
        ("matches/{competition_id}/{season_id}.json", "Matches with scores and teams", "matches/43/106.json (World Cup 2022)"),
        ("events/{match_id}.json", "Every event in a match", "events/3869685.json (2022 final)"),
        ("lineups/{match_id}.json · three-sixty/{match_id}.json", "Lineups / 360 freeze frames", "—"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def main():
    comps = get(f"{BASE}/competitions.json")
    matches = get(f"{BASE}/matches/43/106.json")  # FIFA World Cup 2022
    final = max(matches, key=lambda m: m["match_date"])
    events = get(f'{BASE}/events/{final["match_id"]}.json')
    shots = [e for e in events if e["type"]["name"] == "Shot"]
    xg = Counter()
    for s in shots:
        xg[s["team"]["name"]] += s["shot"]["statsbomb_xg"]
    show({"free_competition_seasons": len(comps), "world_cup_2022_matches": len(matches),
          "final": f'{final["home_team"]["home_team_name"]} {final["home_score"]}-{final["away_score"]} {final["away_team"]["away_team_name"]}',
          "events_in_final": len(events), "event_types": Counter(e["type"]["name"] for e in events).most_common(5),
          "shots_and_xG": {t: {"shots": sum(s["team"]["name"] == t for s in shots), "xG": round(v, 2)} for t, v in xg.items()}})


if __name__ == "__main__":
    main()
