META = dict(
    name="EPA Envirofacts (Toxics Release Inventory and more)",
    domain="Weather & Climate",
    docs="https://www.epa.gov/enviro/envirofacts-data-service-api",
    auth="None",
    signup="No registration. One URL grammar over dozens of EPA databases: TRI (toxic releases), GHGRP (facility "
           "greenhouse-gas emissions), SDWIS (drinking water), RCRA (hazardous waste), and more.",
    provides=["TRI: facilities reporting toxic chemical releases (name, address, county FIPS, industry, coordinates)",
              "GHGRP: annual greenhouse-gas emissions of large facilities", "SDWIS: drinking-water systems and violations"],
    ideas="Environmental justice (toxic releases vs. neighborhood demographics), polluter mapping, compliance studies.",
    endpoint="GET https://data.epa.gov/efservice/{table}/{column}/{value}/rows/{first}:{last}/JSON",
    params=[
        ("table", "Table name, e.g. tri_facility, tri_reporting_form, pub_dim_facility (GHGRP), sdw_contaminant ...", "tri_facility"),
        ("{column}/{value}", "Filter pairs (repeatable; also /{column}/<, >, BEGINNING, CONTAINING)", "state_abbr/DC"),
        ("rows/{first}:{last}", "Row range (0-based)", "rows/0:9"),
        ("format", "JSON | CSV | XML | EXCEL (last path segment)", "JSON"),
        ("count", "Append /COUNT to get the number of matching rows", "/tri_facility/state_abbr/VA/COUNT"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    count = get("https://data.epa.gov/efservice/tri_facility/state_abbr/VA/COUNT/JSON")[0]["TOTALQUERYRESULTS"]
    rows = get("https://data.epa.gov/efservice/tri_facility/state_abbr/VA/fac_closed_ind/0/rows/0:4/JSON")
    show({"VA_TRI_facilities": count, "examples": [
        {k: r.get(k) for k in ("facility_name", "city_name", "county_name", "zip_code")} for r in rows]})


if __name__ == "__main__":
    main()
