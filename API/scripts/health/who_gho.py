META = dict(
    name="WHO Global Health Observatory (GHO OData)",
    domain="Health & Social Science",
    docs="https://www.who.int/data/gho/info/gho-odata-api",
    auth="None",
    signup="No registration.",
    provides=["2,000+ health indicators for 194 countries: life expectancy, mortality, immunization, NCDs, health spending",
              "Breakdowns by sex, age, year"],
    ideas="Global health inequality, health-system efficiency, cross-country panel models.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://ghoapi.azureedge.net/api/WHOSIS_000001", params={  # life expectancy at birth
        "$filter": "SpatialDim in ('USA','JPN','NGA','IND','BRA') and Dim1 eq 'SEX_BTSX' and TimeDim eq 2021"})
    show([{"country": r["SpatialDim"], "year": r["TimeDim"], "life_expectancy": round(r["NumericValue"], 1)}
          for r in d["value"]])


if __name__ == "__main__":
    main()
