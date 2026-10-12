META = dict(
    name="GitHub REST API",
    domain="Engineering, Energy & Tech",
    docs="https://docs.github.com/en/rest",
    auth="None (60 req/hour) / personal access token (5,000 req/hour)",
    env="GITHUB_TOKEN (optional)",
    signup="Optional: GitHub -> Settings -> Developer settings -> Personal access tokens -> Fine-grained -> "
           "public-repo read-only. Pass as `Authorization: Bearer <token>`.",
    provides=["Repository search & stats: stars, forks, language, topics, license, issues",
              "Commits, contributors, pull requests, issues, releases", "User / org profiles"],
    ideas="Open-source ecosystem health, developer-collaboration networks, technology adoption.",
    endpoint="GET https://api.github.com/search/repositories",
    params=[
        ("q", "Keywords + qualifiers: language:, topic:, stars:>N, created:>DATE, user:, org:", "topic:machine-learning language:python"),
        ("sort", "stars | forks | updated | help-wanted-issues", "stars"),
        ("order", "desc | asc", "desc"),
        ("per_page / page", "Max 100 per page; search capped at 1,000 results", "5"),
        ("/repos/{owner}/{repo}", "Sibling endpoint: repo detail (+ /commits, /issues?state=all, /contributors, /releases)", "/repos/pandas-dev/pandas"),
        ("Authorization (header)", "Optional token (5,000 req/hour)", "Bearer <GITHUB_TOKEN>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import os

from _common import get, show


def main():
    hdr = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.getenv("GITHUB_TOKEN") else {}
    d = get("https://api.github.com/search/repositories", headers=hdr, params={
        "q": "topic:machine-learning language:python", "sort": "stars", "per_page": 5})
    show({"total": d["total_count"], "repos": [
        {"repo": r["full_name"], "stars": r["stargazers_count"], "forks": r["forks_count"],
         "open_issues": r["open_issues_count"], "license": (r.get("license") or {}).get("spdx_id")} for r in d["items"]]})


if __name__ == "__main__":
    main()
