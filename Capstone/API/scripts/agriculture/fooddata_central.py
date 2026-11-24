META = dict(
    name="USDA FoodData Central",
    domain="Agriculture & Food",
    docs="https://fdc.nal.usda.gov/api-guide/",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use your api.data.gov key (https://api.data.gov/signup/ or https://fdc.nal.usda.gov/api-key-signup). "
           "1,000 req/hour. Data are public domain (CC0).",
    provides=["Nutrient profiles (energy, protein, fat, vitamins, minerals - 150+ nutrients) for 400k+ foods",
              "Data types: Foundation (lab-analyzed), SR Legacy, Survey (FNDDS, what Americans eat), Branded (labels with UPC)",
              "Ingredients and serving sizes for branded products"],
    ideas="Nutrition scoring and diet optimization, ultra-processed food classification, recipe nutrient estimation.",
    endpoint="GET https://api.nal.usda.gov/fdc/v1/foods/search  ·  GET /fdc/v1/food/{fdcId}",
    params=[
        ("query", "Search text (supports +must -mustnot \"phrases\")", "cheddar cheese"),
        ("dataType", "Foundation | SR Legacy | Survey (FNDDS) | Branded (repeatable)", "Foundation"),
        ("pageSize / pageNumber", "Max 200 per page", "5"),
        ("sortBy / sortOrder", "dataType.keyword | lowercaseDescription.keyword | fdcId | publishedDate; asc | desc", "fdcId / asc"),
        ("brandOwner", "Filter branded foods by owner", "Kraft Heinz"),
        ("/food/{fdcId}?nutrients=", "One food; limit to nutrient numbers", "nutrients=203,204,205"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

NUTRIENTS = {"Energy": "kcal", "Protein": "g", "Total lipid (fat)": "g", "Carbohydrate, by difference": "g"}


def main():
    d = get("https://api.nal.usda.gov/fdc/v1/foods/search", params={
        "query": "cheddar cheese", "dataType": "Foundation", "pageSize": 3,
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")})
    show({"matches": d["totalHits"], "foods": [
        {"fdcId": f["fdcId"], "description": f["description"], "category": f.get("foodCategory"),
         "per_100g": {n["nutrientName"]: f'{n["value"]} {n["unitName"].lower()}' for n in f["foodNutrients"]
                      if n["nutrientName"] in NUTRIENTS and n["unitName"].lower() == NUTRIENTS[n["nutrientName"]]}}
        for f in d["foods"]]})


if __name__ == "__main__":
    main()
