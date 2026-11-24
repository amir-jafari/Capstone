META = dict(
    name="Kenneth French Data Library (Fama-French factors)",
    domain="Finance & Economics",
    docs="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    auth="None (zipped CSV downloads)",
    signup="No registration; no REST API. Zipped CSVs updated monthly from CRSP. Files contain several tables "
           "(monthly, then annual) separated by blank lines and text headers, so parse by section. pandas-datareader "
           "can also read them ('famafrench').",
    provides=["Fama-French 3 and 5 factors (market, size, value, profitability, investment) and momentum, since 1926",
              "Portfolios sorted on size, book-to-market, momentum, industries (5-49), for the US and international regions",
              "Risk-free rate (1-month T-bill)"],
    ideas="Asset-pricing tests, factor investing backtests, explaining mutual-fund returns with factor regressions.",
    endpoint="GET https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{file}_CSV.zip",
    params=[
        ("F-F_Research_Data_Factors", "Monthly + annual 3 factors: Mkt-RF, SMB, HML, RF", "—"),
        ("F-F_Research_Data_5_Factors_2x3", "5 factors: adds RMW, CMA (from 1963)", "—"),
        ("F-F_Momentum_Factor", "Momentum (Mom)", "—"),
        ("49_Industry_Portfolios", "Value- and equal-weighted industry returns", "—"),
        ("units", "Returns in percent; dates YYYYMM (monthly) or YYYY (annual)", "—"),
    ],
)
import csv
import io
import statistics
import zipfile

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def main():
    z = zipfile.ZipFile(io.BytesIO(requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=60).content))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    monthly = [r for r in csv.reader(lines) if r and r[0].strip().isdigit() and len(r[0].strip()) == 6]  # YYYYMM rows
    cols = ["Mkt-RF", "SMB", "HML", "RF"]
    data = [{"month": r[0].strip(), **{c: float(v) for c, v in zip(cols, r[1:5])}} for r in monthly]
    show({"months": len(data), "first": data[0]["month"], "last_3_months": data[-3:],
          "average_monthly_pct_since_1926": {c: round(statistics.mean(d[c] for d in data), 3) for c in cols}})


if __name__ == "__main__":
    main()
