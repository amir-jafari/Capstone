META = dict(
    name="NSF Award Search API",
    domain="Science & Research",
    docs="https://resources.research.gov/common/webapi/awardapisearch-v1.htm",
    auth="None",
    signup="No registration. Every National Science Foundation award since ~1960 with abstracts.",
    provides=["Awards: title, abstract, amount obligated, start/end dates, program, directorate, PI and institution",
              "Search by keyword, institution, state, PI name, program, date range"],
    ideas="Science-funding landscape (which topics/institutions get funded), abstract topic modeling, funding vs. output (OpenAlex).",
    endpoint="GET https://api.nsf.gov/services/v1/awards.json",
    params=[
        ("keyword", "Full-text search in title/abstract (supports quotes)", "\"machine learning\""),
        ("awardeeName", "Institution - wrap in double quotes, otherwise the filter is silently ignored", "\"George Washington University\""),
        ("awardeeStateCode / awardeeCity", "Institution location", "DC"),
        ("dateStart / dateEnd", "Award date range, MM/DD/YYYY", "01/01/2025"),
        ("fundProgramName", "NSF program", "—"),
        ("printFields", "Documented field selector, but currently ignored (full records are returned)", "—"),
        ("rpp / offset", "Results per page (max 25) and 1-based offset", "25 / 1"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.nsf.gov/services/v1/awards.json", params={
        "awardeeName": '"George Washington University"',  # the quotes are required for this filter to apply
        "dateStart": "01/01/2025", "rpp": 25})
    awards = sorted(d["response"]["award"], key=lambda a: -float(a.get("fundsObligatedAmt") or 0))
    show({"gwu_awards_since_2025 (first page of 25)": len(awards), "awardee": awards[0]["awardee"] if awards else None,
          "total_obligated_usd (first page)": round(sum(float(a.get("fundsObligatedAmt") or 0) for a in awards)),
          "largest": [{"title": a["title"], "pi": f'{a.get("piFirstName", "")} {a.get("piLastName", "")}',
                       "amount_usd": a.get("fundsObligatedAmt"), "program": a.get("fundProgramName"),
                       "start": a.get("startDate")} for a in awards[:4]]})


if __name__ == "__main__":
    main()
