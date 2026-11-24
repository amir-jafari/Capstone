META = dict(
    name="US Treasury Fiscal Data",
    domain="Finance & Economics",
    docs="https://fiscaldata.treasury.gov/api-documentation/",
    auth="None",
    signup="No registration.",
    provides=["Debt to the Penny (daily national debt)", "Average interest rates on Treasury securities",
              "Monthly Treasury Statement: receipts, outlays, deficit", "Exchange rates used by Treasury"],
    ideas="Fiscal-policy analysis, debt dynamics, government spending composition.",
    endpoint="GET https://api.fiscaldata.treasury.gov/services/api/fiscal_service/{endpoint}",
    params=[
        ("endpoint", "Dataset path (see the dataset catalog on the docs page)", "v2/accounting/od/debt_to_penny"),
        ("fields", "Comma-separated columns", "record_date,tot_pub_debt_out_amt"),
        ("filter", "field:op:value, ops eq gt gte lt lte in; comma = AND", "record_date:gte:2026-01-01"),
        ("sort", "Column, prefix - for descending", "-record_date"),
        ("page[size] / page[number]", "Paging (default 100)", "5"),
        ("format", "json | csv | xml", "json"),
    ],
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
