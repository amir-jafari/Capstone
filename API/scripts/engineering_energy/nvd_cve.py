META = dict(
    name="NIST National Vulnerability Database (CVE API 2.0)",
    domain="Engineering, Energy & Tech",
    docs="https://nvd.nist.gov/developers/vulnerabilities",
    auth="None (optional key)",
    env="NVD_API_KEY (optional)",
    signup="Keyless: 5 req / 30 s. Request a key at https://nvd.nist.gov/developers/request-an-api-key -> 50 req / 30 s.",
    provides=["Every CVE: description, CVSS scores/severity, CWE weakness type, affected products (CPE), references"],
    ideas="Vulnerability-severity prediction from text, exploit-likelihood modeling, cyber-risk dashboards.",
    endpoint="GET https://services.nvd.nist.gov/rest/json/cves/2.0",
    params=[
        ("keywordSearch / keywordExactMatch", "Words in the description; flag for exact phrase", "pytorch"),
        ("cveId", "A single CVE", "CVE-2021-44228"),
        ("cvssV3Severity", "LOW | MEDIUM | HIGH | CRITICAL", "CRITICAL"),
        ("cweId", "Weakness type", "CWE-79"),
        ("pubStartDate / pubEndDate", "Published range, ISO-8601 (max 120 days)", "2026-01-01T00:00:00.000"),
        ("cpeName", "Affected product (CPE 2.3 name)", "cpe:2.3:a:pytorch:pytorch:*:*:*:*:*:*:*:*"),
        ("resultsPerPage / startIndex", "Max 2,000 per page", "5"),
        ("apiKey (header)", "Optional key", "<NVD_API_KEY>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://services.nvd.nist.gov/rest/json/cves/2.0", params={
        "keywordSearch": "pytorch", "resultsPerPage": 5})
    out = []
    for v in d["vulnerabilities"]:
        c = v["cve"]
        m = (c.get("metrics", {}).get("cvssMetricV31") or [{}])[0].get("cvssData", {})
        out.append({"id": c["id"], "published": c["published"][:10], "severity": m.get("baseSeverity"),
                    "score": m.get("baseScore"), "description": c["descriptions"][0]["value"]})
    show({"total": d["totalResults"], "cves": out}, max_str=140)


if __name__ == "__main__":
    main()
