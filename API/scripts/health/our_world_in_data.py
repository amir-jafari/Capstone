META = dict(
    name="Our World in Data (Grapher API)",
    domain="Health & Social Science",
    docs="https://docs.owid.io/projects/etl/api/chart-api/",
    auth="None",
    signup="No registration. Append `.csv` / `.metadata.json` to any chart URL on ourworldindata.org.",
    provides=["Thousands of curated social-science series: poverty, democracy indices, education, CO2, health, happiness",
              "Clean country-year CSVs with sources documented"],
    ideas="Fast panel data for social science (e.g., democracy vs. growth, happiness vs. income).",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import csv
import io

from _common import get, show


def main():
    text = get("https://ourworldindata.org/grapher/happiness-cantril-ladder.csv",
               params={"useColumnShortNames": "true"}, as_json=False)
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["code"] in ("USA", "FIN", "IND", "BRA", "DEU")]
    latest = max(r["year"] for r in rows)
    show([r for r in rows if r["year"] == latest])


if __name__ == "__main__":
    main()
