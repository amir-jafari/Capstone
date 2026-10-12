META = dict(
    name="HAPI FHIR public test server (synthetic EHR data)",
    domain="Health & Social Science",
    docs="https://hapi.fhir.org/",
    auth="None",
    signup="No registration. A public sandbox of HL7 FHIR R4 - the standard API every certified US EHR exposes. "
           "Data are synthetic/test records posted by developers: never put real patient data here.",
    provides=["FHIR resources: Patient, Observation (labs, vitals), Condition, MedicationRequest, Encounter, Procedure ...",
              "Standard FHIR search parameters, _include/_revinclude, paging bundles",
              "Practice for SMART-on-FHIR apps that later run against real EHRs (Epic, Cerner)"],
    ideas="Learning FHIR before working with real EHR exports; building ETL from FHIR to analysis tables.",
    endpoint="GET https://hapi.fhir.org/baseR4/{ResourceType}",
    params=[
        ("ResourceType", "Patient | Observation | Condition | MedicationRequest | Encounter | ...", "Observation"),
        ("code", "Coded filter (LOINC/SNOMED), system|code", "http://loinc.org|8867-4 (heart rate)"),
        ("patient / subject", "Resources for one patient", "patient=Patient/123"),
        ("date / _lastUpdated", "Date filters with prefixes ge, le, gt, lt", "_lastUpdated=ge2026-01-01"),
        ("_count / _sort / _summary", "Page size, sort field, count-only", "_count=5 / _sort=-_lastUpdated"),
        ("_include", "Pull referenced resources into the bundle", "Observation:patient"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://hapi.fhir.org/baseR4/Observation", params={
        "code": "http://loinc.org|8867-4", "_count": 5, "_sort": "-_lastUpdated"})  # heart rate
    out = []
    for e in d.get("entry", []):
        o = e["resource"]
        q = o.get("valueQuantity", {})
        out.append({"id": o["id"], "code": o["code"]["coding"][0].get("display"), "value": q.get("value"),
                    "unit": q.get("unit"), "subject": (o.get("subject") or {}).get("reference"),
                    "effective": o.get("effectiveDateTime")})
    show({"resourceType": d["resourceType"], "type": d["type"], "heart_rate_observations": out})


if __name__ == "__main__":
    main()
