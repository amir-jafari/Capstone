META = dict(
    name="US Treasury Fiscal Data",
    domain="Finance & Economics",
    docs="https://fiscaldata.treasury.gov/api-documentation/",
    auth="None",
    signup="No registration.",
    provides=["Debt to the Penny (daily national debt)", "Average interest rates on Treasury securities",
              "Monthly Treasury Statement: receipts, outlays, deficit", "Exchange rates used by Treasury"],
    ideas="Fiscal-policy analysis, debt dynamics, government spending composition.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny",
            params={"sort": "-record_date", "page[size]": 5,
                    "fields": "record_date,tot_pub_debt_out_amt,debt_held_public_amt,intragov_hold_amt"})
    show(d["data"])


if __name__ == "__main__":
    main()
