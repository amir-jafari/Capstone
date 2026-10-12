META = dict(
    name="Delphi Epidata (CMU disease surveillance)",
    domain="Health & Social Science",
    docs="https://cmu-delphi.github.io/delphi-epidata/",
    auth="None (free key raises limits)",
    env="DELPHI_API_KEY (optional)",
    signup="Works keyless at low volume. Register at https://api.delphi.cmu.edu/epidata/admin/registration_form for a key. "
           "Python client: `pip install epidatpy`.",
    provides=["CDC FluView: weekly influenza-like illness (ILI) rates nationally, by HHS region and state since 1997",
              "COVIDcast signals (cases, hospital admissions, search trends, wastewater) by county/state",
              "Nowcasts and forecasting archives with data revisions ('as of' versions)"],
    ideas="Flu forecasting (the CDC FluSight challenge), nowcasting with revisions, signal-combination models.",
    endpoint="GET https://api.delphi.cmu.edu/epidata/{source}/",
    params=[
        ("source", "fluview | fluview_clinical | covidcast | nidss_flu | ...", "fluview"),
        ("regions", "nat | hhs1..hhs10 | state codes (fluview)", "nat"),
        ("epiweeks", "YYYYWW, list, or range YYYYWW-YYYYWW", "202530-202540"),
        ("issues / lag", "Get data as it was reported at an issue week, or with a given reporting lag", "issues=202541"),
        ("covidcast: data_source, signal, geo_type, geo_values, time_values", "COVIDcast query fields", "geo_type=state&geo_values=dc"),
        ("api_key", "Optional key", "<DELPHI_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.delphi.cmu.edu/epidata/fluview/", params={"regions": "nat", "epiweeks": "202501-202552"})
    rows = sorted(d["epidata"], key=lambda r: r["epiweek"])
    peak = max(rows, key=lambda r: r["wili"])
    show({"series": "US weighted influenza-like illness (% of visits), 2025", "weeks": len(rows),
          "peak": {"epiweek": peak["epiweek"], "wILI_pct": peak["wili"]},
          "last_4_weeks": [{"epiweek": r["epiweek"], "wILI_pct": r["wili"], "visits": r["num_patients"]} for r in rows[-4:]]})


if __name__ == "__main__":
    main()
