META = dict(
    name="NOAA Space Weather Prediction Center (SWPC) data",
    domain="Science & Research",
    docs="https://services.swpc.noaa.gov/",
    auth="None",
    signup="No registration. Browsable JSON/text files under /json and /products, updated every minute to every few hours.",
    provides=["Planetary K-index (geomagnetic storms), NOAA space-weather scales (G, S, R), alerts and watches",
              "Real-time solar wind (speed, density, magnetic field), X-ray flux and solar flares",
              "Aurora forecasts (OVATION), 27-day outlook, sunspot/solar-cycle progression"],
    ideas="Geomagnetic-storm forecasting, solar-cycle analysis, space-weather impacts on GPS or the power grid.",
    endpoint="GET https://services.swpc.noaa.gov/{json | products}/{file}.json",
    params=[
        ("products/noaa-planetary-k-index.json", "3-hourly Kp for the past week", "—"),
        ("products/noaa-scales.json", "Current and forecast G/S/R scale levels", "—"),
        ("products/alerts.json", "Recent alerts, watches and warnings (text)", "—"),
        ("json/solar-cycle/observed-solar-cycle-indices.json", "Monthly sunspot numbers since 1749", "—"),
        ("json/rtsw/rtsw_wind_1m.json", "Real-time solar wind, 1-minute: proton speed, density, temperature (the old products/solar-wind files are gone)", "—"),
        ("json/goes/primary/xrays-1-day.json", "GOES X-ray flux (flare detection)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://services.swpc.noaa.gov"


def main():
    kp = get(f"{BASE}/products/noaa-planetary-k-index.json")
    cycle = get(f"{BASE}/json/solar-cycle/observed-solar-cycle-indices.json")
    peak = max(kp, key=lambda r: r["Kp"])
    show({"kp_last_week": {"max": peak["Kp"], "at": peak["time_tag"], "storm": peak["Kp"] >= 5},
          "latest_kp": kp[-3:],
          "monthly_sunspot_number": [{"month": r["time-tag"], "ssn": r["ssn"]} for r in cycle[-4:]]})


if __name__ == "__main__":
    main()
