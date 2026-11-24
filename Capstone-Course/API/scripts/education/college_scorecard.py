META = dict(
    name="College Scorecard (US Dept. of Education)",
    domain="Education",
    docs="https://collegescorecard.ed.gov/data/api-documentation/",
    auth="api.data.gov key (`DEMO_KEY` works at low volume)",
    env="DATA_GOV_API_KEY",
    signup="Use your api.data.gov key (https://api.data.gov/signup/). 1,000 req/hour; DEMO_KEY 30/hour.",
    provides=["Every US college: admissions rate, SAT/ACT ranges, enrollment, demographics, tuition & net price",
              "Completion and retention rates, student debt, loan repayment", "Earnings of graduates by institution and by field of study"],
    ideas="College ROI / value-added models, equity in access and outcomes, predicting graduation rates.",
    endpoint="GET https://api.data.gov/ed/collegescorecard/v1/schools.json",
    params=[
        ("school.name / school.state / id", "Filters (id = IPEDS unit id)", "school.state=DC"),
        ("{field}__range", "Numeric range filter", "latest.student.size__range=5000.."),
        ("fields", "Comma-separated dotted field names (see the data dictionary)", "school.name,latest.admissions.admission_rate.overall"),
        ("sort", "field:asc | field:desc", "latest.student.size:desc"),
        ("per_page / page", "Max 100 per page", "5"),
        ("latest.programs.cip_4_digit", "Field-of-study level data (earnings, debt by program)", "fields=latest.programs.cip_4_digit"),
        ("api_key", "api.data.gov key", "DEMO_KEY"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

FIELDS = {
    "school.name": "name",
    "latest.student.size": "undergrads",
    "latest.admissions.admission_rate.overall": "admit_rate",
    "latest.cost.tuition.out_of_state": "tuition_out_of_state",
    "latest.completion.consumer_rate": "completion_rate",
    "latest.earnings.10_yrs_after_entry.median": "median_earnings_10yr",
}


def main():
    d = get("https://api.data.gov/ed/collegescorecard/v1/schools.json", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "school.state": "DC",
        "school.degrees_awarded.predominant": 3, "fields": ",".join(FIELDS),
        "sort": "latest.student.size:desc", "per_page": 5})
    show({"total": d["metadata"]["total"],
          "schools": [{FIELDS[k]: s.get(k) for k in FIELDS} for s in d["results"]]})


if __name__ == "__main__":
    main()
