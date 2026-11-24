META = dict(
    name="OpenAlex (scholarly works)",
    domain="Science & Research",
    docs="https://docs.openalex.org/",
    auth="None (add `mailto=` for the faster 'polite pool')",
    signup="No registration. 100k req/day. Add `mailto=you@gwu.edu` to requests.",
    provides=["250M+ papers: title, authors, institutions, venues, concepts/topics, citations, open-access links",
              "Author / institution / funder entities", "Abstracts (inverted index), citation graph, group-by aggregations"],
    ideas="Science-of-science, citation networks, research-trend forecasting, institution rankings.",
    endpoint="GET https://api.openalex.org/works",
    params=[
        ("search", "Full-text search over title, abstract, fulltext", "large language models"),
        ("filter", "Comma-separated (AND) attribute filters", "from_publication_date:2024-01-01,is_oa:true"),
        ("sort", "cited_by_count | publication_date | relevance_score, with :desc", "cited_by_count:desc"),
        ("per-page / page", "Max 200 per page", "5"),
        ("cursor", "Deep paging beyond 10,000 results (start with *)", "*"),
        ("select", "Return only these fields", "id,title,cited_by_count"),
        ("group_by", "Counts by a field", "publication_year"),
        ("mailto", "Your email → faster polite pool", "you@gwu.edu"),
        ("/authors · /institutions · /sources · /topics · /funders", "Other entity endpoints with the same params", "/institutions?search=george washington"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.openalex.org/works", params={
        "search": "large language models", "filter": "from_publication_date:2024-01-01",
        "sort": "cited_by_count:desc", "per-page": 5, "mailto": "your_email@gwu.edu"})
    show({"total": d["meta"]["count"], "top_cited": [
        {"title": w["title"], "year": w["publication_year"], "cited_by": w["cited_by_count"],
         "venue": (w.get("primary_location") or {}).get("source", {}) and w["primary_location"]["source"].get("display_name")}
        for w in d["results"]]})


if __name__ == "__main__":
    main()
