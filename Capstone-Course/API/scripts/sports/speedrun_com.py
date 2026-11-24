META = dict(
    name="Speedrun.com API",
    domain="Sports",
    docs="https://github.com/speedruncomorg/api",
    auth="None for reading",
    signup="No registration to read (about 100 requests/minute). Data on the speedrunning community for 40,000+ games.",
    provides=["Games, categories, levels, platforms; leaderboards with run times and ranks",
              "Individual runs with video links, verification status, dates, players", "World-record histories over time"],
    ideas="How world records improve over time (learning curves), community growth by game, verification-time analysis.",
    endpoint="GET https://www.speedrun.com/api/v1/{resource}",
    params=[
        ("games?name= · games/{id}", "Find a game; embed=categories adds its categories", "name=super mario 64"),
        ("games/{id}/records?top=", "Top runs for every category", "top=1"),
        ("leaderboards/{game}/category/{category}?top=", "One leaderboard", "—"),
        ("runs?game=&status=verified&orderby=date&direction=desc", "Run history", "—"),
        ("max / offset", "Paging (max 200)", "20"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

API = "https://www.speedrun.com/api/v1"


def main():
    game = get(f"{API}/games", params={"name": "super mario 64", "embed": "categories"})["data"][0]
    records = get(f'{API}/games/{game["id"]}/records', params={"top": 1, "max": 20, "scope": "full-game"})["data"]
    cats = {c["id"]: c["name"] for c in game["categories"]["data"]}
    out = []
    for r in records:
        if r["runs"]:
            run = r["runs"][0]["run"]
            secs = run["times"]["primary_t"]
            out.append({"category": cats.get(r["category"], r["category"]),
                        "world_record": f"{int(secs // 3600)}:{int(secs % 3600 // 60):02d}:{secs % 60:05.2f}",
                        "date": run.get("date")})
    show({"game": game["names"]["international"], "released": game["released"], "world_records": out})


if __name__ == "__main__":
    main()
