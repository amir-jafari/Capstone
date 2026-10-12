META = dict(
    name="DailyMed (NLM drug labels)",
    domain="Health & Social Science",
    docs="https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm",
    auth="None",
    signup="No registration. The official FDA-submitted labels (package inserts) as Structured Product Labeling (SPL).",
    provides=["Current and historical drug labels: indications, dosage, warnings, boxed warnings, adverse reactions",
              "Search by drug name, NDC, RxCUI, manufacturer, pharmacologic class", "Label versions over time and label images"],
    ideas="NLP on boxed warnings and adverse-reaction sections, tracking label changes after safety signals.",
    endpoint="GET https://dailymed.nlm.nih.gov/dailymed/services/v2/{resource}.json",
    params=[
        ("spls.json?drug_name= / ?rxcui= / ?ndc=", "Labels (set IDs) matching a drug", "drug_name=metformin"),
        ("labeler / published_date_comparison", "Filter by manufacturer / publication date", "labeler=Teva"),
        ("pagesize / page", "Paging (max 100)", "5"),
        ("spls/{setid}/history.json", "Versions of one label", "—"),
        ("spls/{setid}.xml", "Full label XML (sections by LOINC code)", "—"),
        ("drugnames.json?drug_name=", "Normalized drug names", "drug_name=lipitor"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json", params={"drug_name": "semaglutide",
                                                                                   "pagesize": 5})
    show({"labels": d["metadata"]["total_elements"],
          "examples": [{"title": s["title"], "published": s["published_date"], "setid": s["setid"]} for s in d["data"]]},
         max_str=140)


if __name__ == "__main__":
    main()
