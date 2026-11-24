META = dict(
    name="Bank for International Settlements (BIS) Data Portal API",
    domain="Finance & Economics",
    docs="https://stats.bis.org/api-doc/v2/",
    auth="None",
    signup="No registration. SDMX REST API over the BIS statistics used by central banks worldwide.",
    provides=["Central-bank policy rates for ~40 economies (daily/monthly, back decades)",
              "Residential property prices, credit to the private sector and credit-to-GDP gaps, debt-service ratios",
              "Effective exchange rates, consumer prices, international banking and debt-securities statistics"],
    ideas="Global monetary-policy cycles, credit booms and housing busts, early-warning indicators of financial crises.",
    endpoint="GET https://stats.bis.org/api/v2/data/dataflow/BIS/{flow}/1.0/{key}",
    params=[
        ("flow", "WS_CBPOL policy rates | WS_SPP property prices | WS_TC total credit | WS_EER effective FX | WS_DSR debt service", "WS_CBPOL"),
        ("key", "Dimension values joined by dots, + for several (WS_CBPOL: FREQ.REF_AREA)", "M.US+GB+XM+JP+BR"),
        ("startPeriod / lastNObservations", "Time filters", "lastNObservations=1"),
        ("format", "csv (jsondata currently returns HTTP 406)", "csv"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    text = get("https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.US+GB+XM+JP+BR+IN+TR",
               params={"lastNObservations": 1, "format": "csv"}, as_json=False)
    show({"central_bank_policy_rates_pct": {r["REF_AREA"]: f'{r["OBS_VALUE"]} ({r["TIME_PERIOD"]})'
                                            for r in csv.DictReader(io.StringIO(text))}})


if __name__ == "__main__":
    main()
