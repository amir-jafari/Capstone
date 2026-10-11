META = dict(
    name="CDC Open Data (Socrata SODA)",
    domain="Health & Social Science",
    docs="https://dev.socrata.com/foundry/data.cdc.gov/swc5-untb",
    auth="None (optional app token raises throttle)",
    env="SOCRATA_APP_TOKEN (optional)",
    signup="Works keyless. For heavy use create an app token at https://data.cdc.gov/profile/edit/developer_settings. "
           "The same SODA query language works on data.cityofchicago.org, data.cityofnewyork.us, opendata.dc.gov, etc.",
    provides=["CDC PLACES: county/tract estimates of obesity, diabetes, depression, smoking, insurance...",
              "Overdose deaths, COVID, vaccination, BRFSS, NCHS mortality", "SQL-like filtering ($where, $select, $group)"],
    ideas="Social determinants of health, small-area health disparities, joining with Census ACS.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    # CDC PLACES county-level data: depression prevalence in DC/MD/VA counties
    d = get("https://data.cdc.gov/resource/swc5-untb.json", params={
        "$select": "stateabbr,locationname,measure,data_value,year", "measureid": "DEPRESSION",
        "datavaluetypeid": "CrdPrv", "$where": "stateabbr in ('DC','MD','VA')",
        "$order": "data_value DESC", "$limit": 5})
    show(d)


if __name__ == "__main__":
    main()
