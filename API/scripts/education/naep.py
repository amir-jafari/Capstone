META = dict(
    name="NAEP — The Nation's Report Card",
    domain="Education",
    docs="https://www.nationsreportcard.gov/api_documentation.aspx",
    auth="None",
    signup="No registration.",
    provides=["National, state and large-district (TUDA) assessment results in math, reading, science, writing, civics, history",
              "Average scale scores, achievement-level percentages, percentiles, gaps by race, gender, school lunch eligibility",
              "Trends back to the 1990s with standard errors"],
    ideas="Pandemic learning-loss analysis, state policy evaluation, achievement-gap trends.",
    endpoint="GET https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx?type=data",
    params=[
        ("subject", "mathematics | reading | science | writing | civics | ushistory | ...", "mathematics"),
        ("grade", "4 | 8 | 12", "8"),
        ("subscale", "Composite scale code: MRPCM (math), RRPCM (reading), SRPUV (science)", "MRPCM"),
        ("variable", "Reporting group: TOTAL, GENDER, SDRACE, SLUNCH3, ...", "TOTAL"),
        ("jurisdiction", "NP (national public), state codes, TUDA districts (comma list)", "NP,DC,MD,VA"),
        ("stattype", "MN:MN (mean), ALC:BB/AB/PR/AD (achievement levels), RP:… (percentiles)", "MN:MN"),
        ("Year", "Assessment year(s), comma list", "2019,2022,2024"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx", params={
        "type": "data", "subject": "mathematics", "grade": 8, "subscale": "MRPCM", "variable": "TOTAL",
        "jurisdiction": "NP,DC,MD,VA", "stattype": "MN:MN", "Year": "2019,2024"})
    show([{"year": r["year"], "jurisdiction": r["jurisLabel"], "grade8_math_avg_score": round(r["value"], 1)}
          for r in d["result"]], max_items=8)


if __name__ == "__main__":
    main()
