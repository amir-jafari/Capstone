META = dict(
    name="openFDA",
    domain="Health & Social Science",
    docs="https://open.fda.gov/apis/",
    auth="None (optional key)",
    env="OPENFDA_API_KEY (optional)",
    signup="Keyless: 1,000 req/day per IP. Free key at https://open.fda.gov/apis/authentication/ -> 120k/day.",
    provides=["Adverse event reports for drugs, devices, food (FAERS)", "Drug labels, recalls/enforcement, NDC directory",
              "`count=` aggregations directly in the API"],
    ideas="Pharmacovigilance / adverse-event signal detection, recall analysis.",
    endpoint="GET https://api.fda.gov/{category}/{endpoint}.json",
    params=[
        ("category/endpoint", "drug/event, drug/label, drug/enforcement, device/event, food/enforcement, ...", "drug/event"),
        ("search", "field:term with AND/OR (+), ranges [A TO B]", "patient.drug.openfda.generic_name:\"semaglutide\""),
        ("count", "Aggregate counts by a field (.exact for whole phrases)", "patient.reaction.reactionmeddrapt.exact"),
        ("limit / skip", "Max 1,000 per request; skip up to 25,000", "6"),
        ("sort", "field:asc|desc", "receivedate:desc"),
        ("api_key", "Optional key", "<OPENFDA_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.fda.gov/drug/event.json", params={
        "search": 'patient.drug.openfda.generic_name:"semaglutide"',
        "count": "patient.reaction.reactionmeddrapt.exact", "limit": 6})
    show({"drug": "semaglutide", "top_reported_reactions": d["results"]}, max_items=6)


if __name__ == "__main__":
    main()
