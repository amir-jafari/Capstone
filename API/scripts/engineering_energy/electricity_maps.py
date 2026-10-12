META = dict(
    name="Electricity Maps API",
    domain="Engineering, Energy & Tech",
    docs="https://app.electricitymaps.com/docs",
    auth="API token (header `auth-token`)",
    env="ELECTRICITYMAPS_TOKEN",
    signup="Create a free account at https://portal.electricitymaps.com - the free tier covers live and recent data for "
           "a single zone for non-commercial use; historical/multi-zone access is paid (academic discounts exist).",
    provides=["Live and historical carbon intensity (gCO2eq/kWh) for 200+ grid zones worldwide",
              "Power breakdown by source incl. imports/exports, renewable and fossil-free percentages", "Forecasts (paid tiers)"],
    ideas="Carbon-aware computing, comparing grid decarbonization across regions, marginal vs. average emissions.",
    endpoint="GET https://api.electricitymap.org/v3/{resource}",
    params=[
        ("carbon-intensity/latest?zone=", "Latest intensity for a zone", "zone=US-MIDA-PJM"),
        ("power-breakdown/latest?zone=", "Latest generation and import mix", "zone=DE"),
        ("carbon-intensity/history?zone=", "Past 24 hours", "zone=GB"),
        ("lat / lon", "Alternative to zone: resolve the zone from coordinates", "38.9 / -77.04"),
        ("/v3/zones", "List of zone codes", "—"),
        ("auth-token (header)", "Your token", "<ELECTRICITYMAPS_TOKEN>"),
    ],
    sample="""{
  "zone": "US-MIDA-PJM",
  "carbonIntensity": 372,
  "datetime": "2026-10-12T12:00:00.000Z",
  "renewablePercentage": 11,
  "fossilFreePercentage": 47
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show

BASE = "https://api.electricitymap.org/v3"


def main():
    hdr = {"auth-token": key("ELECTRICITYMAPS_TOKEN")}
    ci = get(f"{BASE}/carbon-intensity/latest", headers=hdr, params={"zone": "US-MIDA-PJM"})
    pb = get(f"{BASE}/power-breakdown/latest", headers=hdr, params={"zone": "US-MIDA-PJM"})
    show({"zone": ci["zone"], "carbonIntensity": ci["carbonIntensity"], "datetime": ci["datetime"],
          "renewablePercentage": pb.get("renewablePercentage"), "fossilFreePercentage": pb.get("fossilFreePercentage")})


if __name__ == "__main__":
    main()
