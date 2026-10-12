META = dict(
    name="UN Data Portal — World Population Prospects",
    domain="Census & Demographics",
    docs="https://population.un.org/dataportalapi/index.html",
    auth="Bearer token (free)",
    env="UN_POP_TOKEN",
    signup="Email population@un.org (subject 'Data Portal API token request') or follow the link on the docs page; "
           "a bearer token is returned. Pass as `Authorization: Bearer <token>`.",
    provides=["Population by age/sex for every country 1950-2100 (projections)",
              "Fertility, mortality, life expectancy, migration, dependency ratios"],
    ideas="Aging & dependency forecasting, demographic dividend studies.",
    sample="""[
  {"location": "Nigeria", "indicator": "Total population by sex (thousands)", "year": 2025, "value": 237527.8},
  {"location": "Nigeria", "indicator": "Total population by sex (thousands)", "year": 2050, "value": 359427.6}
]""",
    endpoint="GET https://population.un.org/dataportalapi/api/v1/data/indicators/{ids}/locations/{ids}/start/{year}/end/{year}",
    params=[
        ("indicators", "Indicator ID(s); list via /api/v1/indicators", "49 (total population)"),
        ("locations", "UN M49 country codes; list via /api/v1/locations", "566 (Nigeria), 840 (USA)"),
        ("start / end", "Year range (estimates 1950-2023, projections to 2100)", "2025 / 2050"),
        ("pageNumber / pageSize", "Paging", "1 / 100"),
        ("Authorization (header)", "Bearer token", "Bearer <UN_POP_TOKEN>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    # Indicator 49 = total population; location 566 = Nigeria
    d = get("https://population.un.org/dataportalapi/api/v1/data/indicators/49/locations/566/start/2025/end/2050",
            headers={"Authorization": f"Bearer {key('UN_POP_TOKEN')}"})
    show([{"location": r["location"], "indicator": r["indicator"], "year": r["timeLabel"], "value": r["value"]}
          for r in d["data"] if r["sex"] == "Both sexes" and r["timeLabel"] in ("2025", "2050")])


if __name__ == "__main__":
    main()
