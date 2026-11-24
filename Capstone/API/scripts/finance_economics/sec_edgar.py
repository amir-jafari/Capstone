META = dict(
    name="SEC EDGAR (data.sec.gov)",
    domain="Finance & Economics",
    docs="https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
    auth="None (User-Agent with email required)",
    signup="No registration. Must send `User-Agent: Name email@domain`; max 10 req/sec.",
    provides=["Every company filing (10-K, 10-Q, 8-K, S-1...) with dates & links",
              "XBRL 'company facts': every reported financial number (revenue, net income, EPS...) over time",
              "Full-text search (efts.sec.gov) across filings since 2001"],
    ideas="NLP on 10-K risk factors, fraud detection, fundamentals-based models, ESG text mining.",
    endpoint="GET https://data.sec.gov/…",
    params=[
        ("submissions/CIK{10-digit CIK}.json", "Company profile + recent filings list", "CIK0000320193"),
        ("api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json", "One financial concept over time", "CIK0000320193/us-gaap/NetIncomeLoss"),
        ("api/xbrl/companyfacts/CIK{cik}.json", "Every XBRL fact a company has reported", "CIK0000320193"),
        ("api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json", "One concept across all companies for a period", "us-gaap/Revenues/USD/CY2024"),
        ("https://www.sec.gov/files/company_tickers.json", "Ticker → CIK lookup table", "AAPL → 320193"),
        ("User-Agent (header)", "Required: name + email", "GWU Capstone you@gwu.edu"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

HDR = {"User-Agent": "GWU Capstone research your_email@gwu.edu"}


def main():
    sub = get("https://data.sec.gov/submissions/CIK0000320193.json", headers=HDR)  # Apple
    recent = sub["filings"]["recent"]
    facts = get("https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/NetIncomeLoss.json",
                headers=HDR)["units"]["USD"]
    annual = [f for f in facts if f.get("form") == "10-K" and f.get("fp") == "FY"][-3:]
    show({
        "company": sub["name"], "sic": sub["sicDescription"],
        "recent_filings": [{"form": recent["form"][i], "date": recent["filingDate"][i]} for i in range(5)],
        "net_income_10K": [{"fy": f["fy"], "end": f["end"], "USD": f["val"]} for f in annual],
    })


if __name__ == "__main__":
    main()
