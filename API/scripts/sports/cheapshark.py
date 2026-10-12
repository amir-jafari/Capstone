META = dict(
    name="CheapShark (PC game deals)",
    domain="Sports",
    docs="https://apidocs.cheapshark.com/",
    auth="None",
    signup="No registration. Be reasonable with request volume (they throttle heavy use). Covers ~30 stores "
           "(Steam, GOG, Humble, Epic, Fanatical...).",
    provides=["Current deals: sale price, normal price, savings %, Metacritic score, Steam rating, store, release date",
              "Price lookups by game title or Steam app ID, cheapest price ever", "Store list"],
    ideas="Discount strategies and timing, price vs. review score, how quickly game prices decay after release.",
    endpoint="GET https://www.cheapshark.com/api/1.0/{deals | games | stores}",
    params=[
        ("storeID", "Store (1 = Steam; list via /stores)", "1"),
        ("upperPrice / lowerPrice", "Price range in USD", "15"),
        ("sortBy", "Deal Rating | Title | Savings | Price | Metacritic | Reviews | Release | Store | Recent", "Metacritic"),
        ("metacritic / steamRating", "Minimum scores", "80"),
        ("games?title= · games?id=", "Find a game / its deals and cheapest ever price", "title=hades"),
        ("pageSize / pageNumber", "Paging (max 60)", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    deals = get("https://www.cheapshark.com/api/1.0/deals", params={
        "storeID": 1, "upperPrice": 15, "metacritic": 85, "sortBy": "Savings", "pageSize": 6})
    stores = get("https://www.cheapshark.com/api/1.0/stores")
    show({"stores_tracked": sum(s["isActive"] for s in stores), "steam_deals_under_15_metacritic_85_plus": [
        {"title": d["title"], "sale": float(d["salePrice"]), "normal": float(d["normalPrice"]),
         "savings_pct": round(float(d["savings"])), "metacritic": int(d["metacriticScore"]),
         "steam_rating": d.get("steamRatingText")} for d in deals]})


if __name__ == "__main__":
    main()
