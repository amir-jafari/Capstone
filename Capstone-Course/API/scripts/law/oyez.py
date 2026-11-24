META = dict(
    name="Oyez (U.S. Supreme Court cases)",
    domain="Law & Justice",
    docs="https://github.com/walkerdb/supreme_court_transcripts (community notes; no official docs)",
    auth="None",
    signup="No registration. Unofficial API behind oyez.org (Cornell LII / Justia): be gentle and cache results.",
    provides=["Every Supreme Court case: question presented, facts, conclusion, parties, lower court",
              "Decision: majority/minority vote counts, winning party, how each justice voted",
              "Timeline (granted/argued/decided), advocates, oral-argument audio and transcripts"],
    ideas="Predicting Supreme Court outcomes, oral-argument transcript analysis, justice voting blocs.",
    endpoint="GET https://api.oyez.org/cases",
    params=[
        ("filter", "term:{year} (the October Term)", "term:2024"),
        ("per_page", "Results per page; 0 = all cases in the term", "0"),
        ("/cases/{term}/{docket}", "Full case record with decisions, votes, advocates", "/cases/2024/23-191"),
        ("decisions[].votes[]", "Field in the case record: each justice's vote (majority / minority)", "—"),
        ("oral_argument_audio[].href", "Field in the case record: link to the transcript and audio record", "—"),
    ],
)
import re

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    cases = get("https://api.oyez.org/cases", params={"filter": "term:2024", "per_page": 0})
    out = []
    for c in cases[:4]:
        full = get(c["href"])
        dec = (full.get("decisions") or [{}])[0]
        out.append({"case": full["name"], "docket": full["docket_number"].strip(),
                    "question": re.sub("<[^>]+>", "", full.get("question") or "").strip(),  # strip HTML tags
                    "vote": f'{dec["majority_vote"]}-{dec["minority_vote"]}' if dec.get("majority_vote") is not None else None,
                    "winner": dec.get("winning_party")})
    show({"term_2024_cases": len(cases), "examples": out}, max_str=200)


if __name__ == "__main__":
    main()
