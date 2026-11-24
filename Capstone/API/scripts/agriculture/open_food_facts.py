META = dict(
    name="Open Food Facts",
    domain="Agriculture & Food",
    docs="https://openfoodfacts.github.io/openfoodfacts-server/api/",
    auth="None for reading",
    signup="No registration. Send a descriptive User-Agent. Limits: 100 product reads/min, 10 searches/min - for bulk "
           "work download the full database (CSV/JSONL/Parquet) instead. ODbL licence.",
    provides=["3M+ packaged food products worldwide (crowd-sourced): ingredients, nutrition facts, allergens, labels",
              "Nutri-Score, NOVA processing group (1-4), Eco-Score, additives, packaging", "Look up by barcode or search by category, brand, country"],
    ideas="Ultra-processed food analysis, nutrition-label vs. Nutri-Score studies, food-product image classification.",
    endpoint="GET https://world.openfoodfacts.org/api/v2/{product/{barcode} | search}",
    params=[
        ("product/{barcode}", "One product by EAN/UPC barcode", "product/737628064502"),
        ("categories_tags_en / brands_tags / countries_tags_en", "Search filters (English tag names)", "breakfast-cereals"),
        ("nutrition_grades_tags / nova_groups_tags", "Filter by Nutri-Score (a-e) or NOVA group (1-4)", "nova_groups_tags=4"),
        ("fields", "Comma list of fields to return", "product_name,brands,nutriscore_grade,nova_group"),
        ("page / page_size", "Paging", "1 / 5"),
        ("sort_by", "popularity_key (other sort keys exist but returned 503 when tested)", "popularity_key"),
    ],
)
from collections import Counter

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://world.openfoodfacts.org/api/v2/search", params={
        "categories_tags_en": "breakfast-cereals", "countries_tags_en": "united-states", "sort_by": "popularity_key",
        "fields": "product_name,brands,nutriscore_grade,nova_group", "page_size": 100})
    products = d["products"]
    show({"US_breakfast_cereals": d["count"],
          "nutriscore_of_100_most_popular": dict(sorted(Counter(p.get("nutriscore_grade") for p in products).items(), key=str)),
          "nova_group_of_100_most_popular": dict(sorted(Counter(p.get("nova_group") for p in products).items(), key=str)),
          "examples": [{k: p.get(k) for k in ("product_name", "brands", "nutriscore_grade", "nova_group")} for p in products[:3]]})


if __name__ == "__main__":
    main()
