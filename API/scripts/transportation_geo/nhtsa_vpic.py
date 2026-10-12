META = dict(
    name="NHTSA vPIC (vehicle VIN decoder)",
    domain="Transportation & Geospatial",
    docs="https://vpic.nhtsa.dot.gov/api/",
    auth="None",
    signup="No registration. Batch decoding of up to 50 VINs per request via POST DecodeVINValuesBatch.",
    provides=["Decode any VIN: make, model, year, body class, engine, fuel type, drive type, plant, safety features",
              "All makes, models per make/year, manufacturer details, vehicle types"],
    ideas="Enriching crash/registration/insurance data with vehicle attributes; EV adoption studies from VIN registries.",
    endpoint="GET https://vpic.nhtsa.dot.gov/api/vehicles/{operation}",
    params=[
        ("DecodeVinValues/{VIN}?format=json&modelyear=", "Flat key/value decode of one VIN (partial VINs with * allowed)", "DecodeVinValues/1HGCV1F34LA000000"),
        ("DecodeVINValuesBatch/ (POST)", "Body DATA=VIN1,year;VIN2,year (up to 50)", "DATA=5UXWX7C5*BA,2011;..."),
        ("GetModelsForMakeYear/make/{make}/modelyear/{year}", "Models a maker sold in a year; make matches substrings, so filter on Make_Name", "make/tesla/modelyear/2024"),
        ("GetAllMakes", "Every make in the database", "—"),
        ("format", "json | csv | xml", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

FIELDS = ["Make", "Model", "ModelYear", "BodyClass", "FuelTypePrimary", "ElectrificationLevel", "DriveType", "PlantCountry"]


def main():
    vin = get("https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/5YJ3E1EA7KF000000", params={"format": "json"})
    models = get("https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/ford/modelyear/2025",
                 params={"format": "json"})
    show({"decoded_vin": {k: vin["Results"][0][k] for k in FIELDS},
          # make= matches substrings (e.g. "Bradford"), so keep exact matches only
          "ford_2025_models": sorted(m["Model_Name"] for m in models["Results"] if m["Make_Name"] == "FORD")},
         max_items=8)


if __name__ == "__main__":
    main()
