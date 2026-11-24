META = dict(
    name="Redfin Data Center (housing market tracker)",
    domain="Housing & Real Estate",
    docs="https://www.redfin.com/news/data-center/",
    auth="None (TSV downloads)",
    signup="No registration; no REST API. Gzipped TSVs on a public S3 bucket, updated monthly (weekly files also "
           "available). National file ~0.5 MB; metro/county/ZIP/city files are much larger. Cite Redfin.",
    provides=["Median sale & list price, price per sq ft, homes sold, pending sales, new listings, inventory, months of supply",
              "Days on market, sale-to-list ratio, share sold above list, price drops", "By national, state, metro, county, city, ZIP, neighborhood and property type"],
    ideas="Market-cycle detection, inventory vs. price dynamics, mortgage-rate shocks on local markets.",
    endpoint="GET https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/{level}_market_tracker.tsv000.gz",
    params=[
        ("level", "us_national | state | redfin_metro | county | city | zip_code | neighborhood", "us_national"),
        ("PROPERTY_TYPE", "All Residential | Single Family Residential | Condo/Co-op | Townhouse | Multi-Family (2-4 Unit)", "All Residential"),
        ("IS_SEASONALLY_ADJUSTED", "true | false", "false"),
        ("PERIOD_BEGIN / PERIOD_DURATION", "Month start date; 30 = monthly rows (90-day rolling in some files)", "2026-04-01 / 30"),
        ("metrics", "MEDIAN_SALE_PRICE, HOMES_SOLD, INVENTORY, MONTHS_OF_SUPPLY, MEDIAN_DOM, AVG_SALE_TO_LIST, ... (+ _MOM / _YOY)", "—"),
    ],
)
import csv
import gzip
import io

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

URL = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/us_national_market_tracker.tsv000.gz"


def main():
    raw = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=60).content
    rows = csv.DictReader(io.StringIO(gzip.decompress(raw).decode()), delimiter="\t")
    rows = sorted((r for r in rows if r["PROPERTY_TYPE"] == "All Residential" and r["IS_SEASONALLY_ADJUSTED"] == "false"),
                  key=lambda r: r["PERIOD_BEGIN"])
    show([{"month": r["PERIOD_BEGIN"][:7], "median_sale_price": int(float(r["MEDIAN_SALE_PRICE"])),
           "price_yoy_pct": round(100 * float(r["MEDIAN_SALE_PRICE_YOY"]), 1), "homes_sold": int(float(r["HOMES_SOLD"])),
           "inventory": int(float(r["INVENTORY"])), "months_of_supply": round(float(r["MONTHS_OF_SUPPLY"]), 1)}
          for r in rows[-4:]])


if __name__ == "__main__":
    main()
