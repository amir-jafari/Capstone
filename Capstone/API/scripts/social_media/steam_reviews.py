META = dict(
    name="Steam user reviews API",
    domain="Social Media",
    docs="https://partner.steamgames.com/doc/store/getreviews",
    auth="None",
    signup="No registration. Public endpoint behind the Steam store; page through all reviews with the cursor. "
           "Millions of labeled (recommended / not recommended) texts - a ready-made sentiment dataset.",
    provides=["Review text, recommended (yes/no), playtime at review, helpful/funny votes, language, timestamps",
              "Review summary: total positive/negative, review score description",
              "Filters by language, recency, purchase type (Steam vs. key), day range"],
    ideas="Sentiment classification with free labels, review bombing detection, playtime vs. satisfaction.",
    endpoint="GET https://store.steampowered.com/appreviews/{appid}?json=1",
    params=[
        ("appid", "Steam app ID (from the store URL)", "730 (Counter-Strike 2)"),
        ("filter", "recent | updated | all (all = by helpfulness)", "recent"),
        ("language", "english | all | german | ...", "english"),
        ("review_type / purchase_type", "all | positive | negative; all | steam | non_steam_purchase", "negative"),
        ("num_per_page / cursor", "Max 100; start with * and pass back the returned cursor (URL-encoded)", "100 / *"),
        ("day_range", "With filter=all: reviews from the last N days", "30"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://store.steampowered.com/appreviews/1145360", params={  # Hades
        "json": 1, "filter": "recent", "language": "english", "num_per_page": 100, "purchase_type": "all"})
    s, revs = d["query_summary"], d["reviews"]
    show({"game": "Hades (appid 1145360)", "score": s["review_score_desc"], "total_positive": s["total_positive"],
          "total_negative": s["total_negative"],
          "recent_100_recommended_pct": round(100 * sum(r["voted_up"] for r in revs) / len(revs)),
          "examples": [{"recommended": r["voted_up"], "hours_at_review": round(r["author"]["playtime_at_review"] / 60, 1),
                        "text": r["review"].replace("\n", " ")} for r in revs[:3]]}, max_str=140)


if __name__ == "__main__":
    main()
