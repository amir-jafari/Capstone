META = dict(
    name="World Bank Indicators API",
    domain="Finance & Economics",
    docs="https://datahelpdesk.worldbank.org/knowledgebase/articles/889392",
    auth="None",
    signup="No registration.",
    provides=["16,000+ development indicators for 200+ countries since 1960",
              "GDP, poverty, education, health, CO2, internet use, governance (WGI)..."],
    ideas="Cross-country development economics, panel regressions, global inequality studies.",
    endpoint="GET https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}",
    params=[
        ("country codes", "ISO3 codes separated by ;, or all", "USA;CHN;IND"),
        ("indicator", "Indicator code (NY.GDP.PCAP.CD, SP.POP.TOTL, SE.XPD.TOTL.GD.ZS, EN.GHG.CO2.PC.CE.AR5, ...)", "NY.GDP.PCAP.CD"),
        ("format", "json | xml", "json"),
        ("date", "Year or range", "2000:2023"),
        ("mrv", "Most recent N values (instead of date)", "5"),
        ("per_page / page", "Paging (default 50)", "50"),
        ("/v2/indicator?source=2", "Sibling endpoint: browse indicator codes (source 2 = WDI)", "per_page=100"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    meta, rows = get("https://api.worldbank.org/v2/country/USA;CHN;IND;BRA;NGA/indicator/NY.GDP.PCAP.CD",
                     params={"format": "json", "date": "2023", "per_page": 50})
    show([{"country": r["country"]["value"], "year": r["date"], "gdp_per_capita_usd": round(r["value"] or 0)}
          for r in rows])


if __name__ == "__main__":
    main()
