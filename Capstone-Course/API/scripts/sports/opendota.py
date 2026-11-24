META = dict(
    name="OpenDota API (Dota 2 esports & matches)",
    domain="Sports",
    docs="https://docs.opendota.com/",
    auth="None (free tier) / API key for higher limits",
    env="OPENDOTA_API_KEY (optional)",
    signup="Free tier: 60 requests/minute, 2,000/day, no key. Includes a SQL 'explorer' endpoint over pro-match data.",
    provides=["Hero statistics: picks and wins by skill bracket and in pro matches, base attributes",
              "Pro and public matches with full player stats (KDA, gold/xp per minute, items)",
              "Players, teams, leagues; SQL explorer over the pro-match database"],
    ideas="Draft/win prediction from hero picks, meta shifts after patches, esports team rating systems.",
    endpoint="GET https://api.opendota.com/api/{resource}",
    params=[
        ("heroStats", "Every hero with pro_pick, pro_win, pro_ban and bracket pick/win counts", "—"),
        ("proMatches", "Recent pro matches (less_than_match_id for paging)", "—"),
        ("matches/{match_id}", "Full match details", "—"),
        ("explorer?sql=", "SQL over the pro-match database (tables matches, player_matches, ...)", "sql=SELECT COUNT(*) FROM matches"),
        ("teams · leagues · players/{account_id}", "Teams, leagues, player profiles", "—"),
        ("api_key", "Optional key", "<OPENDOTA_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    heroes = get("https://api.opendota.com/api/heroStats")
    pro = [h for h in heroes if h.get("pro_pick", 0) >= 50]
    show({"heroes": len(heroes), "most_picked_in_pro": [
        {"hero": h["localized_name"], "pro_picks": h["pro_pick"], "pro_win_pct": round(100 * h["pro_win"] / h["pro_pick"], 1),
         "pro_bans": h.get("pro_ban")} for h in sorted(pro, key=lambda h: -h["pro_pick"])[:5]],
          "best_pro_win_rate_min_50_picks": [
        {"hero": h["localized_name"], "pro_win_pct": round(100 * h["pro_win"] / h["pro_pick"], 1)}
        for h in sorted(pro, key=lambda h: -h["pro_win"] / h["pro_pick"])[:3]]})


if __name__ == "__main__":
    main()
