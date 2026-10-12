META = dict(
    name="IMF DataMapper API (World Economic Outlook)",
    domain="Finance & Economics",
    docs="https://www.imf.org/external/datamapper/api/help",
    auth="None",
    signup="No registration. Each call returns every country and every year for an indicator (country and period "
           "filters in the URL are currently ignored), so filter locally. Full SDMX access to all IMF datasets is at "
           "api.imf.org.",
    provides=["World Economic Outlook series and forecasts (5 years ahead): real GDP growth, inflation, unemployment, "
              "government debt and balance, current account", "Fiscal Monitor and other IMF datasets; country groups and regions",
              "Indicator, country and region metadata"],
    ideas="Forecast-accuracy studies (IMF forecasts vs. realized), growth/debt dynamics, comparing IMF and World Bank data.",
    endpoint="GET https://www.imf.org/external/datamapper/api/v1/{indicator}",
    params=[
        ("indicator", "NGDP_RPCH real GDP growth, PCPIPCH inflation, LUR unemployment, GGXWDG_NGDP gross debt % GDP, BCA_NGDPD current account", "NGDP_RPCH"),
        ("/api/v1/indicators", "List of all indicator codes", "—"),
        ("/api/v1/countries · /regions · /groups", "Country, region and group codes", "—"),
        ("response", "values -> indicator -> ISO3 country -> {year: value}", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

COUNTRIES = ["USA", "CHN", "DEU", "IND", "BRA", "JPN"]


def main():
    growth = get("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH")["values"]["NGDP_RPCH"]
    infl = get("https://www.imf.org/external/datamapper/api/v1/PCPIPCH")["values"]["PCPIPCH"]
    show({"indicators": "real GDP growth % / inflation % (IMF WEO; 2026-2027 are forecasts)", "data": {
        c: {y: {"gdp_growth": growth.get(c, {}).get(y), "inflation": infl.get(c, {}).get(y)} for y in ("2025", "2026", "2027")}
        for c in COUNTRIES}})


if __name__ == "__main__":
    main()
