META = dict(
    name="NASA POWER (solar & meteorological data)",
    domain="Engineering, Energy & Tech",
    docs="https://power.larc.nasa.gov/docs/services/api/",
    auth="None",
    signup="No registration. Global coverage from satellite and reanalysis data, 1981-present (solar from 1984).",
    provides=["Daily/hourly/monthly solar irradiance (GHI, DNI, diffuse), temperature, humidity, wind at 2/10/50 m, precipitation",
              "Any lat/lon on Earth, or regional grids", "Communities tuned for renewable energy (RE), agriculture (AG), buildings (SB)"],
    ideas="Solar and wind resource assessment, crop-yield models with weather covariates, building energy simulation.",
    endpoint="GET https://power.larc.nasa.gov/api/temporal/{daily | hourly | monthly | climatology}/point",
    params=[
        ("parameters", "Comma list: ALLSKY_SFC_SW_DWN (solar, kWh/m²/day), T2M, RH2M, WS10M, WS50M, PRECTOTCORR, ...", "ALLSKY_SFC_SW_DWN,T2M"),
        ("community", "RE (renewable energy) | AG (agroclimatology) | SB (sustainable buildings)", "RE"),
        ("latitude / longitude", "Point location", "38.9 / -77.04"),
        ("start / end", "YYYYMMDD (daily/hourly) or YYYY (monthly)", "20260101 / 20260930"),
        ("format", "JSON | CSV | ASCII | NETCDF", "JSON"),
        ("/api/temporal/daily/regional", "Sibling endpoint: a lat/lon box (latitude-min, latitude-max, ...)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://power.larc.nasa.gov/api/temporal/monthly/point", params={
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M", "community": "RE", "latitude": 38.9, "longitude": -77.04,
        "start": 2025, "end": 2025, "format": "JSON"})["properties"]["parameter"]
    months = [k for k in d["ALLSKY_SFC_SW_DWN"] if not k.endswith("13")]  # month 13 = annual value
    show({"location": "Washington, DC (2025)", "annual_solar_kWh_m2_day": d["ALLSKY_SFC_SW_DWN"]["202513"],
          "monthly": [{"month": m[4:], "solar_kWh_m2_day": d["ALLSKY_SFC_SW_DWN"][m], "temp_C": d["T2M"][m],
                       "wind_10m_m_s": d["WS10M"][m]} for m in months[::3]]})


if __name__ == "__main__":
    main()
