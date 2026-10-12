META = dict(
    name="OECD Data API — Education at a Glance",
    domain="Education",
    docs="https://gitlab.algobank.oecd.org/public-documentation/dotstat-migration/-/raw/main/OECD_Data_API_documentation.pdf",
    auth="None",
    signup="No registration. SDMX REST API over all OECD datasets (~1,500 dataflows). Find dataflow IDs and dimension codes "
           "in the OECD Data Explorer (https://data-explorer.oecd.org) -> Developer API. Limit: 60 data queries/hour.",
    provides=["Education at a Glance: attainment, enrollment, spending per student, teachers' salaries, class sizes",
              "Labour-market outcomes by education: earnings premiums, employment rates, NEET rates, by field of study",
              "The same API serves every OECD topic (economy, health, environment, PISA-related indicators)"],
    ideas="Returns to education across countries, education spending vs. outcomes, gender gaps in earnings by degree.",
    endpoint="GET https://sdmx.oecd.org/public/rest/data/{agency},{dataflow},{version}/{key}",
    params=[
        ("agency,dataflow,version", "Dataflow identifier from the Data Explorer", "OECD.EDU.IMEP,DSD_EAG_LSO_EA@DF_LSO_EARN_REL_UPPER,1.0"),
        ("key", "One slot per dimension, separated by dots; + for several values, empty = all. Every slot must be present (a short key returns 403)", "USA+FIN._T.Y25T64.ISCED11A_6..........."),
        ("startPeriod / endPeriod", "Year range", "2021"),
        ("format", "csvfilewithlabels | csvfile | jsondata | genericdata", "csvfilewithlabels"),
        ("dimensionAtObservation", "AllDimensions for flat JSON observations", "AllDimensions"),
        ("/public/rest/dataflow/all", "Catalog of all dataflows (send Accept: application/vnd.sdmx.structure+json)", "—"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

DIMS = ["REF_AREA", "SEX", "AGE", "ATTAINMENT_LEV", "EDUCATION_FIELD", "MEASURE", "INCOME", "BIRTH_PLACE", "MIGRATION_AGE",
        "EDU_STATUS", "LABOUR_FORCE_STATUS", "DURATION_UNEMP", "UNIT_MEASURE", "STATISTICAL_OPERATION",
        "WORK_TIME_ARNGMNT", "QUESTIONNAIRE", "FREQ"]


def main():
    want = {"REF_AREA": "USA+DEU+FIN+KOR+GBR+CAN", "SEX": "_T", "AGE": "Y25T64",
            "ATTAINMENT_LEV": "ISCED11A_6+ISCED11A_7_8", "WORK_TIME_ARNGMNT": "FT"}
    key = ".".join(want.get(d, "") for d in DIMS)  # every dimension needs a slot, even if empty
    text = get(f"https://sdmx.oecd.org/public/rest/data/OECD.EDU.IMEP,DSD_EAG_LSO_EA@DF_LSO_EARN_REL_UPPER,1.0/{key}",
               params={"startPeriod": 2021, "format": "csvfilewithlabels"}, as_json=False)
    latest = {}
    for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        k = (r["Reference area"], r["Educational attainment level"])
        if k not in latest or r["TIME_PERIOD"] > latest[k]["TIME_PERIOD"]:
            latest[k] = r
    show({"measure": "Earnings of full-time workers aged 25-64, % of upper-secondary graduates' earnings", "data": [
        {"country": c, "level": lvl, "year": r["TIME_PERIOD"], "relative_earnings_pct": round(float(r["OBS_VALUE"]))}
        for (c, lvl), r in sorted(latest.items())]}, max_items=12)


if __name__ == "__main__":
    main()
