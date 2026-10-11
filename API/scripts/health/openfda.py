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
