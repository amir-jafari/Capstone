META = dict(
    name="Zillow Research Data (ZHVI home values, ZORI rents)",
    domain="Housing & Real Estate",
    docs="https://www.zillow.com/research/data/",
    auth="None (CSV downloads)",
    signup="No registration; there is no public REST API - the research page publishes monthly CSVs. Pick a series "
           "and geography on the page and copy the CSV link. Cite Zillow per its terms.",
    provides=["ZHVI: typical home value by metro, state, county, city, ZIP, neighborhood (monthly since 2000)",
              "ZORI: typical observed market rent", "Inventory, new listings, days on market, sale-to-list ratio, forecasts"],
    ideas="Home-price forecasting, rent burden vs. income (join Census ACS), effect of transit or flood risk on values.",
    endpoint="GET https://files.zillowstatic.com/research/public_csvs/{series}/{Geography}_{series}_….csv",
    params=[
        ("series folder", "zhvi (home values), zori (rents), invt_fs (inventory), ...", "zhvi"),
        ("Geography", "Metro | State | County | City | Zip | Neighborhood", "Metro"),
        ("file name", "Encodes home type, price tier, smoothing, seasonal adjustment", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"),
        ("columns", "RegionID, SizeRank, RegionName, RegionType, StateName, then one column per month", "2026-08-31"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def main():
    rows = list(csv.DictReader(io.StringIO(get(URL, as_json=False))))  # ~4.5 MB
    months = [c for c in rows[0] if c[:2] in ("19", "20")]
    out = []
    for r in rows[:6]:  # rows are sorted by SizeRank: US first, then the largest metros
        now, year_ago = float(r[months[-1]]), float(r[months[-13]])
        out.append({"region": r["RegionName"], "typical_home_value": round(now), "yoy_change_pct": round(100 * (now / year_ago - 1), 1)})
    show({"month": months[-1], "metros": out}, max_items=6)


if __name__ == "__main__":
    main()
