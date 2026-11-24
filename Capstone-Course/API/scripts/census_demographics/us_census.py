META = dict(
    name="US Census Bureau Data API",
    domain="Census & Demographics",
    docs="https://www.census.gov/data/developers/guidance/api-user-guide.html",
    auth="API key (query `key`) — now required",
    env="CENSUS_API_KEY",
    signup="Request a key at https://api.census.gov/data/key_signup.html (emailed; click the activation link). "
           "Free; 500 queries/day per IP without key is no longer honored for most datasets. "
           "Variable lists: https://api.census.gov/data/2023/acs/acs5/variables.html",
    provides=["American Community Survey (ACS 1-yr/5-yr): income, education, race, housing, commuting, language...",
              "Decennial Census counts", "Geographies: nation, state, county, tract, block group, ZIP (ZCTA)",
              "Population estimates, County Business Patterns, international trade"],
    ideas="Demographic covariates for any US spatial study, gentrification, segregation indices, equity analysis.",
    sample="""[
  {"NAME": "District of Columbia", "population": "672079", "median_household_income": "106287"},
  {"NAME": "Maryland", "population": "6161707", "median_household_income": "98461"},
  {"NAME": "Virginia", "population": "8624511", "median_household_income": "87249"}
]""",
    endpoint="GET https://api.census.gov/data/{year}/{dataset}",
    params=[
        ("year / dataset", "Vintage + dataset path: acs/acs5, acs/acs1, dec/pl, pep/population, cbp, ...", "2023/acs/acs5"),
        ("get", "Comma-separated variables (max 50) or group(TABLE)", "NAME,B01003_001E,B19013_001E"),
        ("for", "Target geography with FIPS code or *", "state:11, county:*, tract:*"),
        ("in", "Parent geography (required for counties/tracts within a state)", "state:11 county:001"),
        ("ucgid", "Alternative geography selector by GEOID", "0400000US11"),
        ("key", "Your API key", "<CENSUS_API_KEY>"),
        ("/data/{year}/{dataset}/variables.json", "Sibling endpoint: every variable code and label", "2023/acs/acs5/variables.json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    rows = get("https://api.census.gov/data/2023/acs/acs5", params={
        "get": "NAME,B01003_001E,B19013_001E",  # total population, median household income
        "for": "state:11,24,51", "key": key("CENSUS_API_KEY")})
    header, *data = rows
    show([{"NAME": r[0], "population": r[1], "median_household_income": r[2]} for r in data])


if __name__ == "__main__":
    main()
