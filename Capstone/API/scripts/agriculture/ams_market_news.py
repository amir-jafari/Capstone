META = dict(
    name="USDA AMS Market News (MyMarketNews API)",
    domain="Agriculture & Food",
    docs="https://mymarketnews.ams.usda.gov/mymarketnews-api",
    auth="API key as the Basic-auth username (empty password)",
    env="AMS_API_KEY",
    signup="Create a free account at https://mymarketnews.ams.usda.gov, then generate a key under My Profile -> API Key.",
    provides=["Daily/weekly wholesale and terminal-market prices for livestock, grain, dairy, cotton, fruits and vegetables",
              "Cattle and hog auction reports, boxed-beef cutout values, retail feature prices",
              "1,000+ report series, most with history back to the 2000s"],
    ideas="Price transmission from farm to wholesale to retail, seasonality of produce prices, market-shock detection.",
    endpoint="GET https://marsapi.ams.usda.gov/services/v1.2/reports[/{slug_id}]",
    params=[
        ("/reports", "Catalog of report series with slug_id, title, market type", "—"),
        ("/reports/{slug_id}", "Data for one report (e.g. 2466 = national daily boxed beef cutout)", "2466"),
        ("q", "Filter expression field=value;field=value, incl. date ranges with report_begin_date", "report_begin_date=01/01/2026:10/01/2026"),
        ("allSections=true", "Return every section of multi-part reports", "true"),
        ("auth", "HTTP Basic: key as username, blank password", "(AMS_API_KEY, '')"),
    ],
    sample="""[
  {"report_date": "10/09/2026", "slug_name": "AMS_2466", "report_title": "National Daily Boxed Beef Cutout & Boxed Beef Cuts",
   "current_volume": "112", "choice_600_900_current": "362.18", "select_600_900_current": "339.40"}
]""",
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, key, show


def main():
    r = requests.get("https://marsapi.ams.usda.gov/services/v1.2/reports/2466", auth=(key("AMS_API_KEY"), ""),
                     headers={"User-Agent": USER_AGENT}, params={"q": "report_begin_date=10/01/2026:10/12/2026"}, timeout=60)
    r.raise_for_status()
    show(r.json()["results"][:3])


if __name__ == "__main__":
    main()
