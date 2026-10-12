META = dict(
    name="MedlinePlus Connect (patient education)",
    domain="Health & Social Science",
    docs="https://medlineplus.gov/medlineplus-connect/web-service/",
    auth="None",
    signup="No registration. Max 100 requests/minute per IP; cache responses for 12-24 hours.",
    provides=["Consumer-friendly health topic pages (English/Spanish) matched to ICD-10-CM, SNOMED CT, RxNorm, NDC and LOINC codes",
              "Summaries and links for diagnoses, medications and lab tests"],
    ideas="Linking coded EHR/claims data to plain-language explanations; health-literacy and readability studies.",
    endpoint="GET https://connect.medlineplus.gov/service",
    params=[
        ("mainSearchCriteria.v.cs", "Code system OID: ICD-10-CM 2.16.840.1.113883.6.90, RxNorm ...6.88, LOINC ...6.1, SNOMED ...6.96", "2.16.840.1.113883.6.90"),
        ("mainSearchCriteria.v.c", "The code", "E11.9"),
        ("mainSearchCriteria.v.dn", "Optional display name", "Type 2 diabetes"),
        ("informationRecipient.languageCode.c", "en | es", "en"),
        ("knowledgeResponseType", "application/json | text/xml | text/html", "application/json"),
    ],
)
import html
import re

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://connect.medlineplus.gov/service", params={
        "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.90", "mainSearchCriteria.v.c": "E11.9",
        "knowledgeResponseType": "application/json"})["feed"]
    show([{"title": e["title"]["_value"], "url": e["link"][0]["href"],
           "summary": html.unescape(re.sub("<[^>]+>", " ", e["summary"]["_value"])).split(".")[0].strip() + "."}
          for e in d["entry"]], max_str=220)


if __name__ == "__main__":
    main()
