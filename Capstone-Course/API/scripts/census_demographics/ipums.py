META = dict(
    name="IPUMS API (microdata)",
    domain="Census & Demographics",
    docs="https://developer.ipums.org/docs/v2/",
    auth="API key (header `Authorization`)",
    env="IPUMS_API_KEY",
    signup="Register for the collection you need (e.g. IPUMS USA, CPS, NHGIS) at https://uma.pop.umn.edu/, "
           "then create a key at https://account.ipums.org/api_keys. Python client: `pip install ipumspy`.",
    provides=["Harmonized person-level census & survey microdata (US 1850-today, CPS, international censuses)",
              "NHGIS: aggregate census tables + GIS boundary files for every census since 1790"],
    ideas="Intergenerational mobility, historical migration, labor economics with individual records.",
    sample="""[
  {"name": "us2022a", "description": "2022 ACS"},
  {"name": "us2021a", "description": "2021 ACS"},
  {"name": "us2020a", "description": "2020 ACS"}
]""",
    endpoint="GET https://api.ipums.org/metadata/samples  ·  POST https://api.ipums.org/extracts",
    params=[
        ("collection", "usa | cps | ipumsi | atus | nhgis | ...", "usa"),
        ("version", "API version", "2"),
        ("pageSize / pageNumber", "Paging", "5"),
        ("extract body: samples", "Sample IDs to include", "{\"us2022a\": {}}"),
        ("extract body: variables", "Variables to include", "{\"AGE\": {}, \"SEX\": {}, \"INCTOT\": {}}"),
        ("extract body: dataFormat", "csv | fixed_width | stata | spss", "csv"),
        ("Authorization (header)", "Your API key", "<IPUMS_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.ipums.org/metadata/samples", params={"collection": "usa", "version": 2, "pageSize": 5},
            headers={"Authorization": key("IPUMS_API_KEY")})
    show(d["data"])


if __name__ == "__main__":
    main()
