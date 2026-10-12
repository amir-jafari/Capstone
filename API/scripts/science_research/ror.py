META = dict(
    name="ROR (Research Organization Registry)",
    domain="Science & Research",
    docs="https://ror.readme.io/docs/rest-api",
    auth="None",
    signup="No registration. Persistent IDs for 110k+ universities, labs, hospitals and funders; CC0. From 2026 heavy "
           "users are asked to send a client ID header (free).",
    provides=["Organization names (incl. aliases and other languages), type, country, website, established year",
              "Parent/child relationships, external IDs (GRID, ISNI, Wikidata, Crossref Funder ID)",
              "Affiliation matching: map messy affiliation strings to an organization"],
    ideas="Disambiguating institutions in publication/grant data (OpenAlex, NSF, NIH), building collaboration networks.",
    endpoint="GET https://api.ror.org/v2/organizations",
    params=[
        ("query", "Name search", "george washington university"),
        ("affiliation", "Match a full affiliation string (returns chosen match + score); spell names out - abbreviations like GWU can mismatch", "Department of Statistics, George Washington University"),
        ("filter", "types:education, country.country_code:US, status:active", "types:education"),
        ("page", "20 results per page", "1"),
        ("/v2/organizations/{ror_id}", "One organization", "00y4zzh67"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    aff = "Department of Statistics, George Washington University, Washington DC"
    m = get("https://api.ror.org/v2/organizations", params={"affiliation": aff})
    best = next(i for i in m["items"] if i.get("chosen")) if any(i.get("chosen") for i in m["items"]) else m["items"][0]
    org = best["organization"]
    show({"affiliation_input": aff, "matched": best.get("chosen"),
          "score": best.get("score"), "ror_id": org["id"],
          "name": next(n["value"] for n in org["names"] if "ror_display" in n["types"]),
          "types": org["types"], "established": org.get("established"),
          "external_ids": {e["type"]: e.get("preferred") or e["all"][0] for e in org["external_ids"]}})


if __name__ == "__main__":
    main()
