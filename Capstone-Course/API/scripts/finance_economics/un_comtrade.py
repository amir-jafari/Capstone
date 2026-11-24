META = dict(
    name="UN Comtrade (international trade)",
    domain="Finance & Economics",
    docs="https://comtradedeveloper.un.org/ (developer portal; sign-in required)",
    auth="None for the preview endpoint (max 500 rows); free subscription key for full access",
    env="COMTRADE_API_KEY (optional)",
    signup="The /public/v1/preview endpoints need no key (up to 500 records per call). For the full API register at "
           "https://comtradedeveloper.un.org and subscribe to the free tier (500 calls/day, 100k records per call); send "
           "the key as Ocp-Apim-Subscription-Key.",
    provides=["Bilateral merchandise trade for ~200 countries since 1962: imports/exports by HS product code, value (USD), quantity",
              "Annual and monthly data, re-exports, services trade", "Reference lists for countries (M49 codes) and HS commodities"],
    ideas="Trade-war effects (tariffs), supply-chain dependence (e.g. on China for chips), trade gravity models.",
    endpoint="GET https://comtradeapi.un.org/public/v1/preview/{typeCode}/{freqCode}/{clCode}",
    params=[
        ("typeCode / freqCode / clCode", "C goods | S services; A annual | M monthly; HS | SITC | BEC", "C / A / HS"),
        ("reporterCode / partnerCode", "UN M49 country codes (842 USA, 156 China, 484 Mexico; 0 = World)", "842 / 156"),
        ("period", "Year (or YYYYMM)", "2024"),
        ("cmdCode", "HS code(s), TOTAL, or AG2 for all 2-digit chapters", "AG2"),
        ("flowCode", "M imports | X exports", "M"),
        ("response value fields", "primaryValue (USD), netWgt, qty, cmdCode", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

CHAPTERS = {"85": "Electrical machinery", "84": "Machinery & computers", "95": "Toys & sports goods",
            "94": "Furniture & lighting", "39": "Plastics", "61": "Knitted apparel", "87": "Vehicles", "62": "Apparel"}


def main():
    rows = get("https://comtradeapi.un.org/public/v1/preview/C/A/HS", params={
        "reporterCode": 842, "partnerCode": 156, "period": 2024, "cmdCode": "AG2", "flowCode": "M"})["data"]
    rows = sorted(rows, key=lambda r: -(r["primaryValue"] or 0))
    total = sum(r["primaryValue"] or 0 for r in rows)
    show({"flow": "US imports from China, 2024", "total_bn_usd": round(total / 1e9, 1), "top_hs_chapters": [
        {"hs": r["cmdCode"], "name": CHAPTERS.get(r["cmdCode"], "-"), "bn_usd": round(r["primaryValue"] / 1e9, 1)}
        for r in rows[:6]]})


if __name__ == "__main__":
    main()
