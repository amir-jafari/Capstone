META = dict(
    name="European Central Bank Data Portal API",
    domain="Finance & Economics",
    docs="https://data.ecb.europa.eu/help/api/overview",
    auth="None",
    signup="No registration. SDMX REST: find a series key on data.ecb.europa.eu and swap it into the URL.",
    provides=["Euro reference exchange rates (EXR), key ECB interest rates and money-market rates (FM, EST)",
              "Euro-area inflation (ICP), monetary aggregates and bank lending (BSI, MIR), government debt, balance of payments",
              "Survey of Professional Forecasters, bank lending survey"],
    ideas="Monetary-policy transmission studies, euro exchange-rate forecasting, inflation dynamics across countries.",
    endpoint="GET https://data-api.ecb.europa.eu/service/data/{flow}/{key}",
    params=[
        ("flow", "EXR exchange rates | FM financial markets | ICP inflation | BSI balance sheets | MIR bank rates | EST euro short-term rate", "EXR"),
        ("key", "Dot-separated dimensions; + for several values, empty for all", "D.USD+GBP+JPY.EUR.SP00.A"),
        ("startPeriod / endPeriod / lastNObservations", "Time filters", "lastNObservations=5"),
        ("format", "csvdata | jsondata | structurespecificdata", "csvdata"),
        ("detail", "dataonly | serieskeysonly | full", "dataonly"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://data-api.ecb.europa.eu/service/data"


def main():
    fx = get(f"{BASE}/EXR/D.USD+GBP+JPY+CHF.EUR.SP00.A", params={"lastNObservations": 1, "format": "csvdata"}, as_json=False)
    dfr = get(f"{BASE}/FM/B.U2.EUR.4F.KR.DFR.LEV", params={"lastNObservations": 3, "format": "csvdata"}, as_json=False)
    show({"euro_reference_rates": {r["CURRENCY"]: f'{r["OBS_VALUE"]} ({r["TIME_PERIOD"]})' for r in csv.DictReader(io.StringIO(fx))},
          "deposit_facility_rate_changes": [{"date": r["TIME_PERIOD"], "rate_pct": r["OBS_VALUE"]}
                                            for r in csv.DictReader(io.StringIO(dfr))]})


if __name__ == "__main__":
    main()
