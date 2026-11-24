META = dict(
    name="Riot Games API (League of Legends, Valorant, TFT)",
    domain="Sports",
    docs="https://developer.riotgames.com/apis",
    auth="API key (header `X-Riot-Token`)",
    env="RIOT_API_KEY",
    signup="Log in at https://developer.riotgames.com with a Riot account. The development key expires every 24 hours "
           "(regenerate it on the portal); apply for a personal or production key for longer projects. Dev-key limits: "
           "20 requests/second, 100 per 2 minutes. Routing: platform hosts (na1, euw1, kr) vs. regional (americas, europe, asia).",
    provides=["Ranked ladders (Challenger/Grandmaster/Master) with LP and win/loss", "Match history and full match details (champions, items, timeline)",
              "Account lookups by Riot ID; champion mastery; Valorant and TFT equivalents"],
    ideas="Win prediction from draft and early-game stats, skill-rating dynamics, champion meta analysis.",
    endpoint="GET https://{platform | region}.api.riotgames.com/{game}/{api}/{version}/...",
    params=[
        ("na1 · /lol/league/v4/challengerleagues/by-queue/{queue}", "Challenger ladder for RANKED_SOLO_5x5", "RANKED_SOLO_5x5"),
        ("americas · /riot/account/v1/accounts/by-riot-id/{name}/{tag}", "Look up a player's PUUID", "—"),
        ("americas · /lol/match/v5/matches/by-puuid/{puuid}/ids?count=", "Recent match IDs", "count=20"),
        ("americas · /lol/match/v5/matches/{matchId}", "Full match details (/timeline for minute-by-minute)", "—"),
        ("X-Riot-Token (header)", "Your key", "<RIOT_API_KEY>"),
    ],
    sample="""{
  "queue": "RANKED_SOLO_5x5 (NA Challenger)",
  "players": 300,
  "top": [
    {"lp": 1834, "wins": 412, "losses": 331, "win_pct": 55.4},
    {"lp": 1790, "wins": 389, "losses": 310, "win_pct": 55.7}
  ]
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://na1.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5",
            headers={"X-Riot-Token": key("RIOT_API_KEY")})
    top = sorted(d["entries"], key=lambda e: -e["leaguePoints"])[:5]
    show({"queue": "RANKED_SOLO_5x5 (NA Challenger)", "players": len(d["entries"]), "top": [
        {"lp": e["leaguePoints"], "wins": e["wins"], "losses": e["losses"],
         "win_pct": round(100 * e["wins"] / (e["wins"] + e["losses"]), 1)} for e in top]})


if __name__ == "__main__":
    main()
