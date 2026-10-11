META = dict(
    name="Open States / Plural (US state legislatures)",
    domain="Political Science & Government",
    docs="https://docs.openstates.org/api-v3/",
    auth="API key (header `X-API-KEY`)",
    env="OPENSTATES_API_KEY",
    signup="Register at https://open.pluralpolicy.com/accounts/profile/ -> request API key. Free tier: 500 req/day.",
    provides=["Bills, votes, sponsors for all 50 state legislatures + DC/PR", "State legislators (party, district, contact)",
              "Committees, events"],
    ideas="Policy diffusion across states (e.g., model legislation), state-level polarization.",
    sample="""[
  {"identifier": "HB 1234", "title": "Relating to artificial intelligence systems used in employment decisions",
   "jurisdiction": "Virginia", "latest_action": "Referred to Committee on Labor and Commerce", "date": "2026-01-14"}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://v3.openstates.org/bills", headers={"X-API-KEY": key("OPENSTATES_API_KEY")},
            params={"jurisdiction": "Virginia", "q": "artificial intelligence", "sort": "updated_desc", "per_page": 5})
    show([{"identifier": b["identifier"], "title": b["title"], "jurisdiction": b["jurisdiction"]["name"],
           "latest_action": b["latest_action_description"], "date": b["latest_action_date"]} for b in d["results"]])


if __name__ == "__main__":
    main()
