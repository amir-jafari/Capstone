META = dict(
    name="UK Carbon Intensity API (National Grid ESO)",
    domain="Engineering, Energy & Tech",
    docs="https://carbon-intensity.github.io/api-definitions/",
    auth="None",
    signup="No registration. Half-hourly data for Great Britain since 2017, with 96-hour forecasts.",
    provides=["Carbon intensity of electricity (gCO2/kWh): actual and forecast, with index (very low ... very high)",
              "Generation mix by fuel (gas, wind, solar, nuclear, imports, ...)", "Regional intensity for 14 DNO regions and by postcode"],
    ideas="Load-shifting / carbon-aware scheduling, forecasting grid intensity, wind share vs. intensity analysis.",
    endpoint="GET https://api.carbonintensity.org.uk/{resource}",
    params=[
        ("intensity", "Current half-hour (also /intensity/date/{YYYY-MM-DD}, /intensity/{from}/{to})", "intensity"),
        ("intensity/{from}/fw48h", "48-hour forecast from a time", "intensity/2026-10-12T12:00Z/fw48h"),
        ("generation", "Current generation mix in % (also /generation/{from}/{to})", "generation"),
        ("regional / regional/postcode/{outcode}", "All regions, or one area by outward postcode (full district, e.g. SW1A not SW1)", "regional/postcode/RG10"),
        ("intensity/stats/{from}/{to}", "Max/min/average over a period", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://api.carbonintensity.org.uk"


def main():
    now = get(f"{BASE}/intensity")["data"][0]
    mix = get(f"{BASE}/generation")["data"]
    show({"period": f'{now["from"]} - {now["to"]}', "intensity_gCO2_per_kWh": now["intensity"],
          "generation_mix_pct": {f["fuel"]: f["perc"] for f in sorted(mix["generationmix"], key=lambda f: -f["perc"])}},
         max_items=10)


if __name__ == "__main__":
    main()
