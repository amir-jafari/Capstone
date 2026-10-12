META = dict(
    name="RxNorm / RxNav (NLM drug terminology)",
    domain="Health & Social Science",
    docs="https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html",
    auth="None",
    signup="No registration. Max 20 requests/second per IP.",
    provides=["Normalized names and RxCUI codes for every US clinical drug (ingredient, strength, dose form, brand)",
              "Relationships: brand <-> generic, ingredients, drug classes (via RxClass), NDC codes",
              "Approximate matching of messy drug names (e.g. from prescriptions or notes)"],
    ideas="Cleaning and linking medication data across datasets (Medicare Part D, FAERS, EHR extracts).",
    endpoint="GET https://rxnav.nlm.nih.gov/REST/{resource}.json",
    params=[
        ("drugs.json?name=", "All clinical and branded drugs for a name", "name=metformin"),
        ("rxcui.json?name=", "RxCUI for an exact name", "name=Lipitor"),
        ("approximateTerm.json?term=&maxEntries=", "Fuzzy match for misspelled names", "term=metfromin"),
        ("rxcui/{rxcui}/related.json?tty=", "Related concepts by term type (IN ingredient, BN brand, SCD clinical drug, ...)", "rxcui/153165/related.json?tty=IN"),
        ("rxcui/{rxcui}/ndcs.json", "NDC package codes", "—"),
        ("/REST/rxclass/class/byDrugName.json?drugName=", "RxClass: therapeutic/pharmacologic classes", "drugName=metformin"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    groups = get("https://rxnav.nlm.nih.gov/REST/drugs.json", params={"name": "metformin"})["drugGroup"]["conceptGroup"]
    fuzzy = get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json", params={"term": "metfromin", "maxEntries": 1})
    classes = get("https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json", params={"drugName": "metformin",
                                                                                          "relaSource": "ATC"})
    by_type = {g["tty"]: [p["name"] for p in g.get("conceptProperties", []) if "/" not in p["name"]]  # skip combos
               for g in groups}
    show({"clinical_drugs (SCD)": by_type.get("SCD", [])[:4], "branded_drugs (SBD)": by_type.get("SBD", [])[:3],
          "fuzzy_match_for_'metfromin'": fuzzy["approximateGroup"]["candidate"][0]["rxcui"],
          "ATC_classes": sorted({c["rxclassMinConceptItem"]["className"]
                                 for c in classes["rxclassDrugInfoList"]["rxclassDrugInfo"]})})


if __name__ == "__main__":
    main()
