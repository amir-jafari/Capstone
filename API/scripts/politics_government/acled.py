META = dict(
    name="ACLED (Armed Conflict Location & Event Data)",
    domain="Political Science & Government",
    docs="https://acleddata.com/acled-api-documentation",
    auth="OAuth token (account email + password)",
    env="ACLED_EMAIL, ACLED_PASSWORD",
    signup="Register at https://acleddata.com/register/ with your university email (free academic access). "
           "Then POST your email/password to https://acleddata.com/oauth/token to get a 24-hour bearer token.",
    provides=["Geocoded political violence & protest events worldwide (battles, riots, protests, violence vs civilians)",
              "Date, actors, location (lat/lon, admin areas), fatalities, notes text"],
    ideas="Conflict forecasting, protest diffusion, election violence, spatial econometrics.",
    sample="""[
  {"event_date": "2026-09-28", "event_type": "Protests", "sub_event_type": "Peaceful protest",
   "country": "Kenya", "location": "Nairobi", "fatalities": 0,
   "notes": "On 28 September 2026, students demonstrated outside parliament against tuition increases..."}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import requests

from _common import get, key, show


def main():
    tok = requests.post("https://acleddata.com/oauth/token", data={
        "username": key("ACLED_EMAIL"), "password": key("ACLED_PASSWORD"),
        "grant_type": "password", "client_id": "acled"}, timeout=30).json()["access_token"]
    d = get("https://acleddata.com/api/acled/read", headers={"Authorization": f"Bearer {tok}"},
            params={"country": "Kenya", "limit": 5, "_format": "json"})
    show([{k: e[k] for k in ("event_date", "event_type", "sub_event_type", "country", "location", "fatalities", "notes")}
          for e in d["data"]])


if __name__ == "__main__":
    main()
