META = dict(
    name="PVGIS (EU JRC photovoltaic estimates)",
    domain="Engineering, Energy & Tech",
    docs="https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en",
    auth="None",
    signup="No registration. Max ~30 requests/second. Covers Europe, Africa, most of Asia and the Americas.",
    provides=["Expected PV output for a system anywhere: monthly and yearly energy (kWh), irradiation, losses",
              "Optimal tilt/azimuth, hourly time series, typical meteorological year (TMY), off-grid sizing"],
    ideas="Rooftop-solar economics by location, comparing PVGIS vs. NLR PVWatts estimates.",
    endpoint="GET https://re.jrc.ec.europa.eu/api/v5_3/{tool}",
    params=[
        ("tool", "PVcalc (grid-connected) | seriescalc (hourly) | tmy | SHScalc (off-grid) | MRcalc (monthly radiation)", "PVcalc"),
        ("lat / lon", "Location", "38.9 / -77.04"),
        ("peakpower / loss", "System size in kWp; system losses in %", "4 / 14"),
        ("angle / aspect", "Tilt in degrees; azimuth (0 = south, -90 = east, 90 = west)", "30 / 0"),
        ("optimalangles", "1 -> compute optimal tilt and azimuth", "1"),
        ("outputformat", "json | csv | basic", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://re.jrc.ec.europa.eu/api/v5_3/PVcalc", params={
        "lat": 38.9, "lon": -77.04, "peakpower": 4, "loss": 14, "optimalangles": 1, "outputformat": "json"})
    out, mount = d["outputs"], d["inputs"]["mounting_system"]["fixed"]
    show({"system": "4 kWp, 14% losses, Washington DC", "optimal_tilt_deg": mount["slope"]["value"],
          "optimal_azimuth_deg": mount["azimuth"]["value"], "yearly_kWh": out["totals"]["fixed"]["E_y"],
          "monthly_kWh": {m["month"]: m["E_m"] for m in out["monthly"]["fixed"]}}, max_items=12)


if __name__ == "__main__":
    main()
