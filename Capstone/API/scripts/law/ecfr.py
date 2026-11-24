META = dict(
    name="eCFR API (Code of Federal Regulations)",
    domain="Law & Justice",
    docs="https://www.ecfr.gov/developers/documentation/api/v1",
    auth="None",
    signup="No registration. The eCFR is updated daily; the API also returns any past version of a section.",
    provides=["Full text of all 50 CFR titles (structure, XML) as of any date since 2017",
              "Full-text search with section hierarchy and headings", "Change history: which sections were amended when"],
    ideas="Measuring regulatory growth and complexity by agency, tracking how rules change over time, regulation NLP.",
    endpoint="GET https://www.ecfr.gov/api/{service}/v1/{resource}",
    params=[
        ("search/v1/results?query=", "Full-text search (per_page, page, date, title filters)", "query=artificial intelligence"),
        ("search/v1/count?query=", "Number of matching sections", "query=privacy"),
        ("versioner/v1/titles.json", "All titles with latest amended / issue dates", "—"),
        ("versioner/v1/full/{date}/title-{n}.xml", "Full XML of a title (part=, section= to narrow); date must be <= the title's up_to_date_as_of, and the client must accept gzip (requests does; curl needs --compressed)", "/full/2026-10-08/title-45.xml?part=164"),
        ("versioner/v1/versions/title-{n}.json", "Version history of sections in a title", "title-45.json"),
        ("admin/v1/agencies.json", "Agencies and the CFR references they own", "—"),
    ],
)
import re

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.ecfr.gov/api/search/v1/results", params={"query": "artificial intelligence", "per_page": 50})
    sections = {}  # results are per matching paragraph, so keep one row per section
    for r in d["results"]:
        if r["hierarchy"]["section"]:
            sections.setdefault((r["hierarchy"]["title"], r["hierarchy"]["section"]), r)
    titles = get("https://www.ecfr.gov/api/versioner/v1/titles.json")["titles"]
    show({"matching_paragraphs": d["meta"]["total_count"], "example_sections": [
        {"title": r["hierarchy_headings"]["title"], "section": r["hierarchy_headings"]["section"],
         "heading": re.sub("<[^>]+>", "", r["headings"]["section"] or ""),  # strip search-highlight tags
         "agency_chapter": (r["headings"]["chapter"] or "").strip()}
        for r in list(sections.values())[:5]],
          "recently_amended_titles": [{k: t[k] for k in ("number", "name", "latest_amended_on")}
                                      for t in sorted(titles, key=lambda t: t.get("latest_amended_on") or "", reverse=True)[:3]]})


if __name__ == "__main__":
    main()
