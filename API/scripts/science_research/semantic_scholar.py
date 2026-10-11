META = dict(
    name="Semantic Scholar Academic Graph",
    domain="Science & Research",
    docs="https://api.semanticscholar.org/api-docs/",
    auth="None (shared pool) / free key (header `x-api-key`)",
    env="S2_API_KEY (optional)",
    signup="Works keyless in principle, but the shared pool is often rate-limited (HTTP 429). Request a free key at "
           "https://www.semanticscholar.org/product/api#api-key-form -> 1 req/sec dedicated.",
    provides=["Papers with TLDR summaries, abstracts, citations/references, influential-citation counts",
              "SPECTER2 paper embeddings", "Author h-index & paper lists", "Bulk dataset downloads"],
    ideas="Citation-intent analysis, embedding-based paper clustering, literature-review automation.",
    sample="""[
  {"title": "Attention is All you Need", "year": 2017, "citationCount": 160000,
   "tldr": "A new simple network architecture, the Transformer, based solely on attention mechanisms..."}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    hdr = {"x-api-key": os.environ["S2_API_KEY"]} if os.getenv("S2_API_KEY") else {}
    d = get("https://api.semanticscholar.org/graph/v1/paper/search", headers=hdr, params={
        "query": "attention is all you need", "limit": 3, "fields": "title,year,citationCount,tldr"})
    show([{"title": p["title"], "year": p["year"], "citationCount": p["citationCount"],
           "tldr": (p.get("tldr") or {}).get("text")} for p in d["data"]])


if __name__ == "__main__":
    main()
