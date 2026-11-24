META = dict(
    name="CourtListener (Free Law Project)",
    domain="Law & Justice",
    docs="https://wiki.free.law/c/courtlistener/help/api/rest/v4/rest-api-v47",
    auth="None for search (token for higher limits and most other endpoints)",
    env="COURTLISTENER_TOKEN (optional)",
    signup="Search works anonymously. Create a free account, then copy your token from "
           "https://www.courtlistener.com/profile/api/ (header `Authorization: Token <token>`; 5,000 req/hour).",
    provides=["Millions of court opinions (federal and state, incl. the full Caselaw Access Project history) with full text",
              "PACER dockets and filings from the RECAP archive, oral-argument audio, judges and their careers",
              "Citation graph: which opinions cite which (citeCount, cited opinions)"],
    ideas="Legal NLP (outcome prediction, topic models), citation-network influence of precedents, judge behavior.",
    endpoint="GET https://www.courtlistener.com/api/rest/v4/search/",
    params=[
        ("q", "Query; supports \"phrases\", AND/OR, field searches like caseName:(...)", "\"qualified immunity\""),
        ("type", "o opinions | r RECAP dockets | d dockets | p judges | oa oral arguments", "o"),
        ("court", "Space-separated court IDs", "scotus ca4 dcd"),
        ("filed_after / filed_before", "Date range YYYY-MM-DD", "2020-01-01"),
        ("order_by", "score desc | dateFiled desc | citeCount desc", "citeCount desc"),
        ("cursor", "Paging cursor from the 'next' URL (20 results per page)", "—"),
        ("/api/rest/v4/opinions/{id}/", "Sibling endpoint (token): an opinion's full text (plain_text, html_with_citations)", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.courtlistener.com/api/rest/v4/search/", params={
        "q": '"qualified immunity"', "type": "o", "court": "scotus", "order_by": "citeCount desc"})
    show({"scotus_opinions_mentioning_qualified_immunity": d["count"], "most_cited": [
        {"case": r["caseName"], "date": r["dateFiled"], "cited_by": r["citeCount"], "citation": r["citation"][:1],
         "url": "https://www.courtlistener.com" + r["absolute_url"]} for r in d["results"][:5]]})


if __name__ == "__main__":
    main()
