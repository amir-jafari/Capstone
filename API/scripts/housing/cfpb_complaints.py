META = dict(
    name="CFPB Consumer Complaint Database",
    domain="Housing & Real Estate",
    docs="https://cfpb.github.io/api/ccdb/api.html",
    auth="None",
    signup="No registration. 5M+ complaints about mortgages, credit reports, debt collection, banks, student loans...",
    provides=["Each complaint: product, sub-product, issue, company, state, ZIP, date, company response, timeliness",
              "Consumer narrative text (when the consumer consented)", "Built-in aggregations by issue, company, state, product"],
    ideas="NLP on complaint narratives, ranking mortgage servicers, linking complaints to local economic stress.",
    endpoint="GET https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
    params=[
        ("product / sub_product", "Product filter (repeatable)", "Mortgage"),
        ("state / zip_code / company", "Location or company filter (repeatable)", "DC"),
        ("date_received_min / date_received_max", "Date range YYYY-MM-DD", "2025-01-01"),
        ("search_term", "Full-text search across complaint fields, including narratives", "foreclosure"),
        ("has_narrative", "true -> only complaints with published narratives", "true"),
        ("size / frm / sort", "Page size, offset, created_date_desc | relevance_desc", "5 / 0 / created_date_desc"),
        ("format", "json | csv (with no_aggs=true for bulk export)", "csv"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/", params={
        "product": "Mortgage", "state": "DC", "date_received_min": "2025-01-01", "size": 3, "sort": "created_date_desc"})
    issues = d["aggregations"]["issue"]["issue"]["buckets"]
    show({"mortgage_complaints_DC_since_2025": d["hits"]["total"]["value"],
          "top_issues": [{"issue": b["key"], "complaints": b["doc_count"]} for b in issues[:4]],
          "latest": [{k: h["_source"][k] for k in ("date_received", "company", "issue", "company_response")}
                     for h in d["hits"]["hits"]]})


if __name__ == "__main__":
    main()
