META = dict(
    name="Google Fact Check Tools API",
    domain="News & Media",
    docs="https://developers.google.com/fact-check/tools/api",
    auth="API key (query `key`)",
    env="GOOGLE_FACTCHECK_API_KEY",
    signup="In https://console.cloud.google.com create a project, enable 'Fact Check Tools API', and create an API key. Free.",
    provides=["Fact-checks published with ClaimReview markup by outlets like PolitiFact, Snopes, AFP, Reuters, FactCheck.org",
              "Claim text, who made it, date, and each reviewer's rating ('False', 'Misleading'...) with a link",
              "Search by text, filter by publisher site, language and age"],
    ideas="Misinformation datasets, comparing ratings across fact-checkers, claim-matching and verdict classification models.",
    endpoint="GET https://factchecktools.googleapis.com/v1alpha1/claims:search",
    params=[
        ("query", "Text to search claims for", "vaccine"),
        ("languageCode", "BCP-47 language", "en-US"),
        ("reviewPublisherSiteFilter", "Only one fact-checker's site", "politifact.com"),
        ("maxAgeDays", "Only recent fact-checks", "90"),
        ("pageSize / pageToken", "Paging", "10"),
        ("key", "Your API key", "<GOOGLE_FACTCHECK_API_KEY>"),
    ],
    sample="""[
  {"claim": "Example viral claim about vaccines...", "claimant": "Social media posts", "date": "2026-09-28",
   "publisher": "PolitiFact", "rating": "False", "url": "https://www.politifact.com/factchecks/2026/..."}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params={
        "query": "vaccine", "languageCode": "en-US", "maxAgeDays": 90, "pageSize": 5,
        "key": key("GOOGLE_FACTCHECK_API_KEY")})
    show([{"claim": c["text"], "claimant": c.get("claimant"), "date": (c.get("claimDate") or "")[:10],
           "publisher": c["claimReview"][0]["publisher"].get("name"), "rating": c["claimReview"][0].get("textualRating"),
           "url": c["claimReview"][0]["url"]} for c in d.get("claims", [])])


if __name__ == "__main__":
    main()
