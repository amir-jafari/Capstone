META = dict(
    name="USGS Water Services (streamflow)",
    domain="Engineering, Energy & Tech",
    docs="https://waterservices.usgs.gov/docs/instantaneous-values/",
    auth="None",
    signup="No registration. (USGS is migrating to https://api.waterdata.usgs.gov — same data, OGC format.)",
    provides=["Real-time & historical streamflow (discharge), gage height, water temperature, water quality",
              "10,000+ monitoring sites with 15-minute readings"],
    ideas="Flood forecasting, hydrologic time-series ML, drought monitoring.",
    endpoint="GET https://waterservices.usgs.gov/nwis/iv/  (instantaneous)  ·  /nwis/dv/  (daily)",
    params=[
        ("sites", "Comma list of site numbers", "01646500 (Potomac at Little Falls)"),
        ("stateCd / bBox / huc", "Alternative major filters (one required)", "va"),
        ("parameterCd", "00060 discharge (cfs), 00065 gage height (ft), 00010 water temp (°C)", "00060,00065"),
        ("period", "ISO-8601 duration back from now", "PT2H, P7D"),
        ("startDT / endDT", "Absolute range (instead of period)", "2026-10-01"),
        ("siteStatus", "active | inactive | all", "active"),
        ("format", "json | rdb (tab-delimited)", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://waterservices.usgs.gov/nwis/iv/", params={
        "format": "json", "sites": "01646500", "parameterCd": "00060,00065", "period": "PT2H"})  # Potomac @ Little Falls
    show([{"site": ts["sourceInfo"]["siteName"], "variable": ts["variable"]["variableName"],
           "latest": ts["values"][0]["value"][-1]} for ts in d["value"]["timeSeries"]])


if __name__ == "__main__":
    main()
