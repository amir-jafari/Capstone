META = dict(
    name="HUD User API (Fair Market Rents, Income Limits, ZIP crosswalk)",
    domain="Housing & Real Estate",
    docs="https://www.huduser.gov/portal/dataset/fmr-api.html",
    auth="Access token (header `Authorization: Bearer`)",
    env="HUD_API_TOKEN",
    signup="Register at https://www.huduser.gov/hudapi/public/register, log in, and create an access token. Free.",
    provides=["Fair Market Rents (FMR) by bedroom count for every metro and county; Small Area FMRs by ZIP",
              "Income limits (very low / extremely low / low income) by household size",
              "USPS ZIP crosswalk: ZIP <-> tract, county, CBSA, congressional district with address-share ratios"],
    ideas="Rent burden and voucher policy studies, joining ZIP-level data to Census tracts via the crosswalk.",
    endpoint="GET https://www.huduser.gov/hudapi/public/{dataset}",
    params=[
        ("fmr/statedata/{state}", "FMRs for every metro and county in a state", "fmr/statedata/DC"),
        ("fmr/data/{entityid}", "FMRs for one area (metro code or 10-digit county code)", "fmr/data/METRO47900M47900"),
        ("il/data/{entityid}", "Income limits for one area", "il/data/METRO47900M47900"),
        ("usps?type=&query=", "ZIP crosswalk; type 1 ZIP-tract, 2 ZIP-county, 3 ZIP-CBSA, ... query = ZIP, or All", "type=1&query=20052"),
        ("year", "Fiscal year (default latest)", "2026"),
        ("Authorization (header)", "Bearer token", "Bearer <HUD_API_TOKEN>"),
    ],
    sample="""[
  {"metro_name": "Washington-Arlington-Alexandria, DC-VA-MD HUD Metro FMR Area", "code": "METRO47900M47900",
   "Efficiency": 1855, "One-Bedroom": 1906, "Two-Bedroom": 2167, "Three-Bedroom": 2687, "Four-Bedroom": 3192}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://www.huduser.gov/hudapi/public/fmr/statedata/DC",
            headers={"Authorization": f"Bearer {key('HUD_API_TOKEN')}"})["data"]
    show([{k: m.get(k) for k in ("metro_name", "code", "Efficiency", "One-Bedroom", "Two-Bedroom", "Three-Bedroom",
                                 "Four-Bedroom")} for m in d["metroareas"]])


if __name__ == "__main__":
    main()
