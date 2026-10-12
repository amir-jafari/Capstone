META = dict(
    name="EPA ECHO (Enforcement & Compliance History)",
    domain="Weather & Climate",
    docs="https://echo.epa.gov/tools/web-services",
    auth="None",
    signup="No registration. Two-step pattern: get_facilities returns a QueryID, then get_qid pages through the facilities.",
    provides=["800k+ regulated facilities: Clean Air Act, Clean Water Act, hazardous waste (RCRA), drinking water",
              "Compliance status, quarters in non-compliance, inspections, formal enforcement actions, penalties",
              "Facility location, NAICS industry, permits"],
    ideas="Which facilities violate repeatedly, enforcement intensity by neighborhood demographics, penalty deterrence.",
    endpoint="GET https://echodata.epa.gov/echo/echo_rest_services.get_facilities  ->  get_qid",
    params=[
        ("p_st / p_zip / p_ct", "State, ZIP or city filter", "p_st=DC"),
        ("p_act", "Y -> active facilities only", "Y"),
        ("p_med", "Program: A air (CAA), W water (CWA), R hazardous waste (RCRA), S drinking water", "W"),
        ("output", "JSON | XML", "JSON"),
        ("get_qid?qid=&pagesize=&pageno=", "Second call: facility rows for the QueryID", "qid=687&pagesize=5"),
        ("Results summary fields", "QueryRows, INSPRows, FEARows (formal actions), TotalPenalties", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://echodata.epa.gov/echo/echo_rest_services"


def main():
    summary = get(f"{BASE}.get_facilities", params={"output": "JSON", "p_st": "DC", "p_act": "Y"})["Results"]
    page = get(f"{BASE}.get_qid", params={"output": "JSON", "qid": summary["QueryID"], "pagesize": 5})["Results"]
    show({"active_DC_facilities": summary["QueryRows"], "inspections": summary["INSPRows"],
          "formal_enforcement_actions": summary["FEARows"], "total_penalties": summary["TotalPenalties"],
          "examples": [{k: f.get(k) for k in ("FacName", "FacZip", "FacComplianceStatus", "FacQtrsWithNC")}
                       for f in page["Facilities"]]})


if __name__ == "__main__":
    main()
