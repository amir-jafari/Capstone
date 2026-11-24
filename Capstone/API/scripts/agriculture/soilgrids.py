META = dict(
    name="ISRIC SoilGrids (global soil properties)",
    domain="Agriculture & Food",
    docs="https://rest.isric.org/soilgrids/v2.0/docs",
    auth="None",
    signup="No registration. Fair use: ~5 calls/minute (the service is a beta). Global 250 m predictions with uncertainty.",
    provides=["Soil properties anywhere on Earth at 6 depths (0-200 cm): organic carbon, pH, clay/sand/silt, nitrogen, bulk density, CEC",
              "Mean, median and 5/95% quantiles (uncertainty)", "WRB soil classification at a point"],
    ideas="Soil covariates for global crop or carbon models, soil-carbon sequestration potential, land suitability.",
    endpoint="GET https://rest.isric.org/soilgrids/v2.0/properties/query",
    params=[
        ("lon / lat", "Point (WGS84)", "-93.6 / 42.0"),
        ("property", "soc | phh2o | clay | sand | silt | nitrogen | bdod | cec | cfvo | ocd | ocs (repeatable)", "soc"),
        ("depth", "0-5cm | 5-15cm | 15-30cm | 30-60cm | 60-100cm | 100-200cm (repeatable)", "0-5cm"),
        ("value", "mean | Q0.05 | Q0.5 | Q0.95 | uncertainty (repeatable)", "mean"),
        ("unit conversion", "Divide by unit_measure.d_factor to get target units (e.g. pH*10 -> pH)", "—"),
        ("/classification/query?lon=&lat=", "Sibling endpoint: most probable WRB soil class", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://rest.isric.org/soilgrids/v2.0/properties/query", params=[
        ("lon", -93.6), ("lat", 42.0), ("property", "soc"), ("property", "phh2o"), ("property", "clay"),
        ("depth", "0-5cm"), ("depth", "30-60cm"), ("value", "mean")])
    out = {}
    for layer in d["properties"]["layers"]:
        f, unit = layer["unit_measure"]["d_factor"], layer["unit_measure"]["target_units"]
        out[f'{layer["name"]} ({unit})'] = {dep["label"]: dep["values"]["mean"] / f for dep in layer["depths"]}
    show({"point": "-93.6, 42.0 (central Iowa)", "properties": out})


if __name__ == "__main__":
    main()
