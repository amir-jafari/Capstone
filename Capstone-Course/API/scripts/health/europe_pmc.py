META = dict(
    name="Europe PMC",
    domain="Health & Social Science",
    docs="https://europepmc.org/RestfulWebService",
    auth="None",
    signup="No registration. Includes PubMed plus preprints, patents, guidelines and 6M+ open-access full texts.",
    provides=["Search 45M+ life-science abstracts and full texts; citation counts; open-access and preprint flags",
              "Full-text XML for open-access articles", "Text-mined annotations: genes, diseases, chemicals, organisms"],
    ideas="Full-text mining (methods, datasets used), preprint-to-publication tracking, citation analysis.",
    endpoint="GET https://www.ebi.ac.uk/europepmc/webservices/rest/search",
    params=[
        ("query", "Search syntax with fields: TITLE:, ABSTRACT:, AUTH:, PUB_YEAR:, OPEN_ACCESS:y, SRC:PPR (preprints)", "\"long covid\" AND PUB_YEAR:2025 AND OPEN_ACCESS:y"),
        ("format", "json | xml | dc", "json"),
        ("resultType", "idlist | lite | core (core adds abstract, MeSH, full-text links)", "lite"),
        ("pageSize / cursorMark", "Max 1,000; cursorMark=* then nextCursorMark", "5 / *"),
        ("sort", "Field and order, e.g. CITED desc | P_PDATE_D desc", "CITED desc"),
        ("/{source}/{id}/fullTextXML", "Sibling endpoint: open-access full text", "/PMC9000000/fullTextXML"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
        "query": '"long covid" AND PUB_YEAR:2024', "format": "json", "resultType": "lite",
        "pageSize": 5, "sort": "CITED desc"})
    show({"hits": d["hitCount"], "most_cited": [
        {"title": r["title"], "journal": r.get("journalTitle"), "year": r.get("pubYear"),
         "cited_by": r.get("citedByCount"), "open_access": r.get("isOpenAccess")} for r in d["resultList"]["result"]]})


if __name__ == "__main__":
    main()
