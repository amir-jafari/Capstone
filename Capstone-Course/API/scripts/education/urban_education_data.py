META = dict(
    name="Urban Institute Education Data Portal",
    domain="Education",
    docs="https://educationdata.urban.org/documentation/",
    auth="None",
    signup="No registration. Cleaned, harmonized federal education data (CCD, IPEDS, CRDC, EDFacts, Scorecard, SAIPE). "
           "R/Stata packages: `educationdata`.",
    provides=["K-12: every public school and district — enrollment by race/grade, teachers, free/reduced lunch, finance",
              "Civil Rights Data Collection: discipline, AP/IB access, chronic absenteeism", "Higher ed (IPEDS): admissions, enrollment, aid, graduation, finance"],
    ideas="School segregation, discipline disparities, funding vs. outcomes, district-level panel studies.",
    endpoint="GET https://educationdata.urban.org/api/v1/{level}/{source}/{topic}/{year}/[{subtopic}]/",
    params=[
        ("level", "schools | school-districts | college-university", "schools"),
        ("source", "ccd | crdc | edfacts | ipeds | scorecard | saipe | nhgis", "ccd"),
        ("topic / subtopic", "e.g. directory, enrollment/grade-{n}/race, finance, admissions-enrollment, grad-rates", "enrollment/2022/grade-99"),
        ("year", "School year (fall)", "2022"),
        ("fips / leaid / ncessch / unitid", "Filters: state FIPS, district ID, school ID, college ID", "fips=11"),
        ("/api/v1/{…}/summaries?var=&stat=&by=", "Server-side aggregation (sum, avg, count...) grouped by a variable", "var=enrollment&stat=sum&by=fips"),
        ("page / per_page", "Results paged (default 10,000; follow the next URL)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://educationdata.urban.org/api/v1/school-districts/ccd/directory/2022/", params={"fips": 11})
    rows = sorted(d["results"], key=lambda r: -(r.get("enrollment") or 0))
    show({"count": d["count"], "largest_DC_districts": [
        {k: r.get(k) for k in ("lea_name", "enrollment", "number_of_schools", "teachers_total_fte")} for r in rows]})


if __name__ == "__main__":
    main()
