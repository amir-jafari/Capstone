META = dict(
    name="Inside Airbnb (short-term rental data)",
    domain="Transportation & Geospatial",
    docs="https://insideairbnb.com/get-the-data/",
    auth="None (CSV downloads)",
    signup="No registration; no REST API. Quarterly scraped snapshots for 100+ cities. File URLs contain the scrape "
           "date, so read the current links from the get-the-data page. CC BY 4.0; check the site's data policies.",
    provides=["Listings: price, room type, neighbourhood, coordinates, host, minimum nights, availability, reviews per month",
              "Detailed listings (70+ columns incl. amenities, review scores), daily calendar prices, review texts",
              "Neighbourhood boundaries (GeoJSON)"],
    ideas="Short-term rentals vs. housing affordability (with Zillow/ACS), price prediction, review-text NLP, tourism geography.",
    endpoint="GET https://data.insideairbnb.com/{country}/{region}/{city}/{date}/{visualisations | data}/{file}",
    params=[
        ("visualisations/listings.csv", "Summary listings (18 columns)", "united-states/dc/washington-dc/2026-09-19"),
        ("data/listings.csv.gz", "Detailed listings (70+ columns)", "—"),
        ("data/calendar.csv.gz", "Daily price and availability for the next year", "—"),
        ("data/reviews.csv.gz", "Review texts with dates", "—"),
        ("visualisations/neighbourhoods.geojson", "Neighbourhood polygons", "—"),
    ],
)
import csv
import io
import re
from collections import Counter
from statistics import median

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    page = get("https://insideairbnb.com/get-the-data/", as_json=False)
    url = re.search(r"https://data\.insideairbnb\.com/united-states/dc/washington-dc/[\d-]+/visualisations/listings\.csv",
                    page).group(0)  # the newest snapshot link for DC
    rows = list(csv.DictReader(io.StringIO(get(url, as_json=False))))
    prices = [float(r["price"]) for r in rows if r["price"]]
    hoods = Counter(r["neighbourhood"] for r in rows)
    show({"snapshot": url.split("/")[-3], "listings": len(rows), "median_nightly_price_usd": median(prices),
          "by_room_type": Counter(r["room_type"] for r in rows).most_common(),
          "top_neighbourhoods": hoods.most_common(4)})


if __name__ == "__main__":
    main()
