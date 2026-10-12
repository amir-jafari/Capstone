META = dict(
    name="ESCO API (European skills & occupations taxonomy)",
    domain="Jobs & Labor",
    docs="https://esco.ec.europa.eu/en/use-esco/use-esco-services-api",
    auth="None",
    signup="No registration. The EU's multilingual classification (28 languages) of ~3,000 occupations and ~14,000 skills, "
           "mapped to ISCO-08. Complements O*NET for international work.",
    provides=["Occupations with descriptions, ISCO codes, alternative labels in 28 languages",
              "Essential and optional skills/knowledge for each occupation, skill hierarchies and relations",
              "Search across occupations and skills"],
    ideas="Skill extraction from job ads (map phrases to ESCO skills), cross-country occupation matching, skill-gap analysis.",
    endpoint="GET https://ec.europa.eu/esco/api/{search | resource/occupation | resource/skill}",
    params=[
        ("text", "Search text", "data scientist"),
        ("type", "occupation | skill | concept", "occupation"),
        ("language", "Label language (en, de, fr, es, ...)", "en"),
        ("limit / offset", "Paging", "5"),
        ("resource/occupation?uri=", "Full record incl. hasEssentialSkill / hasOptionalSkill links", "uri=http://data.europa.eu/esco/occupation/..."),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://ec.europa.eu/esco/api"


def main():
    hits = get(f"{BASE}/search", params={"text": "data scientist", "type": "occupation", "language": "en", "limit": 3})
    top = hits["_embedded"]["results"][0]
    occ = get(f"{BASE}/resource/occupation", params={"uri": top["uri"], "language": "en"})
    skills = occ["_links"].get("hasEssentialSkill", [])
    show({"matches": hits["total"], "top_match": occ["preferredLabel"]["en"], "isco_code": occ.get("code"),
          "essential_skills": len(skills), "examples": [s["title"] for s in skills[:8]]}, max_items=8)


if __name__ == "__main__":
    main()
