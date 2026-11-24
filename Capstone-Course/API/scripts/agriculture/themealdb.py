META = dict(
    name="TheMealDB (recipes)",
    domain="Agriculture & Food",
    docs="https://www.themealdb.com/api.php",
    auth="Test key `1` in the URL path (free for development)",
    env="THEMEALDB_KEY (optional)",
    signup="The test key 1 works for development and education. Supporters get a production key with extra endpoints "
           "(multiple filters, latest meals, full list).",
    provides=["~600 recipes: name, category, cuisine (area), instructions, ingredients with measures, image, YouTube link",
              "Filter by main ingredient, category or cuisine; random meal; ingredient and category lists"],
    ideas="Cuisine similarity networks from shared ingredients, recipe recommendation, ingredient-image datasets.",
    endpoint="GET https://www.themealdb.com/api/json/v1/{key}/{endpoint}.php",
    params=[
        ("search.php?s= / ?f=", "Search by name, or by first letter", "s=Arrabiata"),
        ("lookup.php?i=", "Full recipe by ID", "i=52772"),
        ("filter.php?i= / ?c= / ?a=", "Meals by main ingredient, category or area (cuisine)", "a=Italian"),
        ("random.php", "One random meal", "—"),
        ("list.php?c=list / ?a=list / ?i=list", "All categories / areas / ingredients", "a=list"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    base = f'https://www.themealdb.com/api/json/v1/{os.getenv("THEMEALDB_KEY", "1")}'
    areas = [a["strArea"] for a in get(f"{base}/list.php", params={"a": "list"})["meals"]]
    italian = get(f"{base}/filter.php", params={"a": "Italian"})["meals"]
    meal = get(f"{base}/lookup.php", params={"i": italian[0]["idMeal"]})["meals"][0]
    ingredients = [f'{meal[f"strMeasure{i}"].strip()} {meal[f"strIngredient{i}"]}'.strip()
                   for i in range(1, 21) if (meal.get(f"strIngredient{i}") or "").strip()]
    show({"cuisines": len(areas), "italian_recipes": len(italian),
          "example": {"name": meal["strMeal"], "category": meal["strCategory"], "ingredients": ingredients,
                      "instructions": meal["strInstructions"]}}, max_items=6, max_str=200)


if __name__ == "__main__":
    main()
