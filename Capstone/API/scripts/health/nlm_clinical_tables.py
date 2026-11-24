META = dict(
    name="NLM Clinical Table Search Service (ICD-10-CM, LOINC, conditions)",
    domain="Health & Social Science",
    docs="https://clinicaltables.nlm.nih.gov/",
    auth="None",
    signup="No registration. Built for autocomplete, so it is fast and returns compact arrays.",
    provides=["ICD-10-CM diagnosis codes, ICD-11, LOINC lab tests, SNOMED-based conditions, RxTerms drugs, HCPCS, NPI",
              "Search by words or code prefix; choose returned fields"],
    ideas="Mapping free-text diagnoses to ICD codes, building code lookups for claims or EHR data.",
    endpoint="GET https://clinicaltables.nlm.nih.gov/api/{table}/v3/search",
    params=[
        ("table", "icd10cm | icd11_codes | loinc_items | conditions | rxterms | hcpcs | npi_idv | ...", "icd10cm"),
        ("terms", "Search text (words or code prefix)", "type 2 diabetes"),
        ("sf", "Fields to search", "code,name"),
        ("df / ef", "Fields to display / extra fields to return", "df=code,name"),
        ("maxList / count / offset", "Max results (up to 500), paging", "maxList=7"),
        ("response shape", "[total, codes, extra_fields, display_rows]", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    total, codes, _, rows = get("https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
                                params={"sf": "code,name", "terms": "type 2 diabetes", "maxList": 7})
    show({"matching_ICD10CM_codes": total, "examples": [{"code": c, "name": r[1]} for c, r in zip(codes, rows)]},
         max_items=7)


if __name__ == "__main__":
    main()
