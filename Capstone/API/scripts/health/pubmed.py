META = dict(
    name="PubMed (NCBI E-utilities)",
    domain="Health & Social Science",
    docs="https://www.ncbi.nlm.nih.gov/books/NBK25501/",
    auth="None (optional free key raises the limit)",
    env="NCBI_API_KEY (optional)",
    signup="Works keyless at 3 requests/second. Create an NCBI account and generate a key under Account Settings for "
           "10 req/s. Add tool= and email= parameters to identify yourself.",
    provides=["39M+ biomedical citations: title, authors, journal, date, MeSH terms, abstracts (efetch)",
              "Search with the full PubMed query language (field tags, MeSH, date ranges, publication types)",
              "The same E-utilities reach Gene, Protein, Nucleotide, ClinVar, SRA and 30+ other NCBI databases"],
    ideas="Biomedical NLP on abstracts, research-trend analysis by MeSH topic, literature-based discovery.",
    endpoint="GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch | esummary | efetch | elink}.fcgi",
    params=[
        ("db", "pubmed | pmc | gene | protein | clinvar | ...", "pubmed"),
        ("term (esearch)", "Query with field tags: [mh] MeSH, [tiab] title/abstract, [dp] date, [pt] publication type", "\"long covid\"[tiab] AND 2025[dp]"),
        ("retmax / retstart / sort", "Page size (max 10,000), offset; sort=pub_date | relevance", "5 / 0 / pub_date"),
        ("id (esummary, efetch)", "Comma-separated PMIDs from esearch", "39811234,39811230"),
        ("rettype / retmode (efetch)", "abstract + text, or xml for full records", "abstract / text"),
        ("usehistory", "y -> keep results server-side for large batch downloads (WebEnv + query_key)", "y"),
        ("retmode / api_key / tool / email", "json | xml; optional key; identify your tool", "json"),
    ],
)
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def main():
    common = {"db": "pubmed", "retmode": "json", "tool": "gwu-capstone-catalog"}
    if os.getenv("NCBI_API_KEY"):
        common["api_key"] = os.environ["NCBI_API_KEY"]
    s = get(f"{BASE}/esearch.fcgi", params={**common, "term": '"long covid"[tiab] AND 2025[dp]', "retmax": 5,
                                            "sort": "pub_date"})["esearchresult"]
    summ = get(f"{BASE}/esummary.fcgi", params={**common, "id": ",".join(s["idlist"])})["result"]
    show({"articles_2025": s["count"], "latest": [
        {"pmid": i, "title": summ[i]["title"], "journal": summ[i]["source"], "pubdate": summ[i]["pubdate"]}
        for i in summ["uids"]]})


if __name__ == "__main__":
    main()
