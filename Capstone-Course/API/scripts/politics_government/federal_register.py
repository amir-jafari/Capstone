META = dict(
    name="Federal Register API",
    domain="Political Science & Government",
    docs="https://www.federalregister.gov/developers/documentation/api/v1",
    auth="None",
    signup="No registration.",
    provides=["Every rule, proposed rule, notice, and presidential document (executive orders) since 1994",
              "Agency, type, abstract, publication date, full-text/PDF links, comment deadlines"],
    ideas="Regulatory-activity indices by administration, executive-order text analysis.",
    endpoint="GET https://www.federalregister.gov/api/v1/documents.json",
    params=[
        ("conditions[term]", "Full-text search", "artificial intelligence"),
        ("conditions[type][]", "RULE | PRORULE | NOTICE | PRESDOCU", "PRORULE"),
        ("conditions[agencies][]", "Agency slug", "environmental-protection-agency"),
        ("conditions[presidential_document_type]", "executive_order | memorandum | proclamation | ...", "executive_order"),
        ("conditions[publication_date][gte] / [lte]", "Date range YYYY-MM-DD", "2025-01-20"),
        ("fields[]", "Fields to return (repeatable)", "title, abstract, document_number"),
        ("order", "newest | oldest | relevance | executive_order_number", "newest"),
        ("per_page / page", "Max 1,000 per page", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.federalregister.gov/api/v1/documents.json", params={
        "conditions[presidential_document_type]": "executive_order", "order": "newest", "per_page": 5,
        "fields[]": ["title", "executive_order_number", "signing_date", "publication_date"]})
    show({"count": d["count"], "results": d["results"]})


if __name__ == "__main__":
    main()
