META = dict(
    name="NLR (formerly NREL) Developer Network — solar, EV charging",
    domain="Engineering, Energy & Tech",
    docs="https://developer.nlr.gov/docs/",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Sign up at https://developer.nlr.gov/signup/ (api.data.gov key). 1,000 req/hour.",
    provides=["PVWatts: estimated solar PV output for any location/system", "Solar resource (GHI/DNI) data",
              "Alternative-fuel / EV charging stations (location, connectors, network)", "Utility rates"],
    ideas="EV-charger access equity, rooftop-solar potential mapping, techno-economic analysis.",
    sample="""{
  "station_counts_near_DC": 1210,
  "stations": [
    {"station_name": "Union Station Garage", "ev_network": "ChargePoint Network",
     "ev_level2_evse_num": 12, "ev_dc_fast_num": null, "distance_mi": 0.6}
  ]
}""",
    endpoint="GET https://developer.nlr.gov/api/alt-fuel-stations/v1/nearest.json",
    params=[
        ("latitude / longitude or location", "Coordinates or a free-text address", "38.9 / -77.04"),
        ("radius", "Miles (default 5; or infinite)", "5"),
        ("fuel_type", "ELEC | E85 | CNG | LPG | HY | BD | LNG | RD (comma list)", "ELEC"),
        ("ev_network", "Charging network", "Tesla"),
        ("ev_charging_level", "1 | 2 | dc_fast", "dc_fast"),
        ("status", "E (open) | P (planned) | T (temporarily unavailable)", "E"),
        ("limit", "Max results", "5"),
        ("/api/pvwatts/v8.json", "Sibling endpoint: solar PV output (lat, lon, system_capacity, azimuth, tilt, array_type, module_type, losses)", "system_capacity=4&tilt=20"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    d = get("https://developer.nlr.gov/api/alt-fuel-stations/v1/nearest.json", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "latitude": 38.9, "longitude": -77.04,
        "fuel_type": "ELEC", "limit": 5})
    show({"station_counts_near_DC": d["total_results"], "stations": [
        {k: s[k] for k in ("station_name", "ev_network", "ev_level2_evse_num", "ev_dc_fast_num")} |
        {"distance_mi": round(s["distance"], 2)} for s in d["fuel_stations"]]})


if __name__ == "__main__":
    main()
