META = dict(
    name="U.S. Drought Monitor",
    domain="Weather & Climate",
    docs="https://droughtmonitor.unl.edu/DmData/DataDownload/WebServiceInfo.aspx",
    auth="None",
    signup="No registration. Weekly maps since 2000 from the National Drought Mitigation Center (UNL), USDA and NOAA. "
           "Send Accept: application/json (otherwise CSV/XML).",
    provides=["Weekly % of area (or population) in each drought category: D0 abnormally dry ... D4 exceptional",
              "National, state, county, HUC watershed, climate-region levels", "Shapefiles/GeoJSON of the weekly maps"],
    ideas="Drought impacts on agriculture, wildfire, migration or electricity (hydropower); county panel studies.",
    endpoint="GET https://usdmdataservices.unl.edu/api/{level}/{statistic}",
    params=[
        ("level", "USStatistics | StateStatistics | CountyStatistics | HUCStatistics | ...", "StateStatistics"),
        ("statistic", "GetDroughtSeverityStatisticsByAreaPercent | ...ByArea | ...ByPopulationPercent | GetDSCI", "GetDroughtSeverityStatisticsByAreaPercent"),
        ("aoi", "Area: state FIPS (two digits, not the abbreviation), county FIPS (five digits), or US", "51 (Virginia)"),
        ("startdate / enddate", "M/D/YYYY", "9/1/2026 / 10/1/2026"),
        ("statisticsType", "1 = cumulative (D0-D4 includes worse), 2 = categorical", "1"),
        ("Accept (header)", "application/json for JSON", "application/json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByAreaPercent",
            headers={"Accept": "application/json"},
            params={"aoi": "51", "startdate": "8/1/2026", "enddate": "10/12/2026", "statisticsType": 1})
    show([{"week": w["mapDate"][:10], **{c: w[c] for c in ("none", "d0", "d1", "d2", "d3", "d4")}}
          for w in sorted(d, key=lambda w: w["mapDate"])[-5:]])


if __name__ == "__main__":
    main()
