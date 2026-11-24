META = dict(
    name="World Bank EdStats (education statistics)",
    domain="Education",
    docs="https://datatopics.worldbank.org/education/",
    auth="None",
    signup="No registration. Same API as the World Bank Indicators entry, but source=12 selects the Education Statistics "
           "database (~4,000 indicators, including learning outcomes and projections). Last refreshed June 2024.",
    provides=["Enrollment, completion, out-of-school rates, pupil-teacher ratios, education spending for every country",
              "Learning outcomes from international assessments (PISA, TIMSS, PIRLS) and learning poverty", "Attainment projections to 2050"],
    ideas="Cross-country education panels, learning poverty vs. income, school spending efficiency.",
    endpoint="GET https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}?source=12",
    params=[
        ("source", "12 = Education Statistics (EdStats)", "12"),
        ("indicator", "EdStats indicator code (browse via /v2/sources/12/indicators)", "SE.TER.ENRR (tertiary gross enrollment)"),
        ("country codes", "ISO3 codes separated by ;, or all", "USA;FIN;KOR;BRA;IND"),
        ("date / mrv", "Year range, or most recent N values (EdStats is sparse - a range plus local filtering works best)", "2010:2023"),
        ("format / per_page", "json | xml; page size", "json / 100"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

INDICATORS = {"SE.TER.ENRR": "tertiary_gross_enrollment_pct", "SE.SEC.ENRR": "secondary_gross_enrollment_pct",
              "SE.XPD.TOTL.GD.ZS": "education_spending_pct_GDP"}


def main():
    out = {}
    for code, label in INDICATORS.items():
        _, rows = get(f"https://api.worldbank.org/v2/country/USA;FIN;KOR;BRA;IND/indicator/{code}",
                      params={"source": 12, "format": "json", "date": "2010:2023", "per_page": 500})
        # EdStats is sparse: walk oldest -> newest and keep each country's latest non-empty value
        for r in sorted(rows or [], key=lambda r: r["date"]):
            if r["value"] is not None:
                out.setdefault(r["country"]["value"], {})[label] = f'{round(r["value"], 1)} ({r["date"]})'
    show(out)


if __name__ == "__main__":
    main()
