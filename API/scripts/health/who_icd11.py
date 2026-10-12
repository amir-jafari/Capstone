META = dict(
    name="WHO ICD-11 API",
    domain="Health & Social Science",
    docs="https://icd.who.int/docs/icd-api/APIDoc-Version2/",
    auth="OAuth2 client credentials (free)",
    env="ICD_CLIENT_ID, ICD_CLIENT_SECRET",
    signup="Register at https://icd.who.int/icdapi -> View API access key(s) for a client ID and secret. Exchange them "
           "for a 1-hour token at https://icdaccessmanagement.who.int/connect/token (scope icdapi_access). "
           "A Docker image is available to run the API locally without limits.",
    provides=["ICD-11 (the current WHO classification, in use since 2022) and ICD-10: codes, titles, definitions, synonyms",
              "Search and autocoding of free text to ICD-11 codes", "Hierarchy navigation, linearizations (MMS for mortality/morbidity)"],
    ideas="Automated coding of clinical text or death certificates, mapping ICD-10 to ICD-11.",
    endpoint="GET https://id.who.int/icd/release/11/{release}/mms/{search | autocode | codeinfo/{code}}",
    params=[
        ("release", "Release ID", "2024-01"),
        ("q (search)", "Search text", "type 2 diabetes"),
        ("searchText (autocode)", "Free text to map to the best code", "heart attack"),
        ("flatResults / highlightingEnabled", "Flat list; turn off <em> highlighting", "true / false"),
        ("headers", "Authorization: Bearer <token>, Accept: application/json, Accept-Language: en, API-Version: v2", "—"),
    ],
    sample="""{
  "query": "type 2 diabetes",
  "matches": [
    {"code": "5A11", "title": "Type 2 diabetes mellitus"},
    {"code": "JA63.2", "title": "Type 2 diabetes mellitus in pregnancy"}
  ]
}""",
)
import re

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    token = requests.post("https://icdaccessmanagement.who.int/connect/token", timeout=30, data={
        "client_id": key("ICD_CLIENT_ID"), "client_secret": key("ICD_CLIENT_SECRET"),
        "scope": "icdapi_access", "grant_type": "client_credentials"}).json()["access_token"]
    d = get("https://id.who.int/icd/release/11/2024-01/mms/search", params={
        "q": "type 2 diabetes", "flatResults": "true", "highlightingEnabled": "false"}, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/json", "Accept-Language": "en", "API-Version": "v2"})
    show({"query": "type 2 diabetes", "matches": [
        {"code": e["theCode"], "title": re.sub("<[^>]+>", "", e["title"])} for e in d["destinationEntities"][:5]]})


if __name__ == "__main__":
    main()
