META = dict(
    name="NHTSA Recalls, Complaints & Safety Ratings",
    domain="Transportation & Geospatial",
    docs="https://www.nhtsa.gov/nhtsa-datasets-and-apis",
    auth="None",
    signup="No registration.",
    provides=["Recalls by vehicle: campaign number, component, summary, consequence, remedy",
              "Owner complaints: crash/fire/injury flags, component, free-text description",
              "NCAP 5-star crash-test ratings (overall, frontal, side, rollover)"],
    ideas="NLP on complaint narratives to detect defects early, recall timing analysis, safety-rating vs. crash outcomes.",
    endpoint="GET https://api.nhtsa.gov/{resource}",
    params=[
        ("recalls/recallsByVehicle?make=&model=&modelYear=", "Recalls for a vehicle", "make=honda&model=accord&modelYear=2020"),
        ("complaints/complaintsByVehicle?make=&model=&modelYear=", "Owner complaints for a vehicle", "make=tesla&model=model 3&modelYear=2022"),
        ("SafetyRatings/modelyear/{year}/make/{make}/model/{model}", "Returns vehicle variants with VehicleId", "modelyear/2024/make/toyota/model/camry"),
        ("SafetyRatings/VehicleId/{id}", "Star ratings for one variant", "VehicleId/19000"),
        ("recalls/campaignNumber?campaignNumber=", "One recall campaign", "campaignNumber=20V682000"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    q = {"make": "tesla", "model": "model 3", "modelYear": 2022}
    recalls = get("https://api.nhtsa.gov/recalls/recallsByVehicle", params=q)["results"]
    complaints = get("https://api.nhtsa.gov/complaints/complaintsByVehicle", params=q)["results"]
    variants = get("https://api.nhtsa.gov/SafetyRatings/modelyear/2022/make/tesla/model/model%203")["Results"]
    rating = get(f'https://api.nhtsa.gov/SafetyRatings/VehicleId/{variants[0]["VehicleId"]}')["Results"][0]
    show({"vehicle": "2022 Tesla Model 3",
          "recalls": [{"campaign": r["NHTSACampaignNumber"], "component": r["Component"]} for r in recalls],
          "complaints": len(complaints),
          "top_complaint_components": Counter(c["components"] for c in complaints).most_common(4),
          "ncap": {k: rating.get(k) for k in ("VehicleDescription", "OverallRating", "OverallFrontCrashRating",
                                              "OverallSideCrashRating", "RolloverRating")}})


if __name__ == "__main__":
    main()
