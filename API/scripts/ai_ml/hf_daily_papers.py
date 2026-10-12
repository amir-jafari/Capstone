META = dict(
    name="Hugging Face Daily Papers",
    domain="AI & Machine Learning",
    docs="https://huggingface.co/docs/hub/en/api",
    auth="None",
    signup="No registration.",
    provides=["Curated daily list of new AI papers (arXiv) with upvotes, comment counts, AI-generated summary, keywords",
              "Linked models, datasets and Spaces for each paper", "Paper search by keyword"],
    ideas="Tracking what the AI community pays attention to, predicting paper popularity from abstracts.",
    endpoint="GET https://huggingface.co/api/daily_papers",
    params=[
        ("limit", "Number of papers", "5"),
        ("date", "A specific day, YYYY-MM-DD", "2026-10-09"),
        ("sort", "publishedAt | trending", "trending"),
        ("/api/papers/search?q=", "Sibling endpoint: search papers", "q=diffusion"),
        ("/api/papers/{arxiv_id}", "Sibling endpoint: one paper's metadata", "/api/papers/1706.03762"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://huggingface.co/api/daily_papers", params={"limit": 5})
    show([{"arxiv_id": p["paper"]["id"], "title": p["paper"]["title"], "upvotes": p["paper"].get("upvotes"),
           "comments": p.get("numComments"), "published": p["paper"]["publishedAt"][:10],
           "summary": p["paper"].get("summary")} for p in d], max_str=200)


if __name__ == "__main__":
    main()
