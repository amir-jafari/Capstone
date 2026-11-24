META = dict(
    name="OpenFEMA — Flood Insurance Claims & Disaster Declarations",
    domain="Housing & Real Estate",
    docs="https://www.fema.gov/about/openfema/api",
    auth="None",
    signup="No registration. Note: the older v2 claims endpoint (FimaNfipClaims) is deprecated and removed after "
           "Oct 15, 2026 - use v3 NfipClaims as below.",
    provides=["2.7M+ National Flood Insurance Program claims: date of loss, flood zone, building/contents payouts, cause, county",
              "NFIP policies, disaster declarations (every federal disaster since 1953), housing-assistance grants by ZIP",
              "OData-style filtering, sorting, selection and counts"],
    ideas="Climate risk and home values, flood-zone mapping accuracy, disaster recovery and housing assistance equity.",
    endpoint="GET https://www.fema.gov/api/open/v3/NfipClaims",
    params=[
        ("$filter", "OData filter (eq, gt, lt, and, or); add amountPaidOnBuildingClaim gt 0 before sorting by payout (nulls sort first)", "state eq 'VA' and yearOfLoss eq 2024"),
        ("$select", "Columns to return", "dateOfLoss,countyCode,ratedFloodZone,amountPaidOnBuildingClaim"),
        ("$orderby", "Sort (desc / asc)", "amountPaidOnBuildingClaim desc"),
        ("$top / $skip", "Paging (max 10,000 per call)", "5"),
        ("$count", "true -> include the total match count in metadata", "true"),
        ("/api/open/v2/DisasterDeclarationsSummaries", "Sibling dataset: every disaster declaration (state, incidentType, dates)", "$filter=state eq 'DC'"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.fema.gov/api/open/v3/NfipClaims", params={
        "$filter": "state eq 'VA' and yearOfLoss eq 2024 and amountPaidOnBuildingClaim gt 0",  # skip unpaid claims
        "$orderby": "amountPaidOnBuildingClaim desc",
        "$select": "dateOfLoss,countyCode,ratedFloodZone,amountPaidOnBuildingClaim,amountPaidOnContentsClaim",
        "$top": 5, "$count": "true"})
    show({"VA_paid_claims_2024": d["metadata"]["count"], "largest_building_payouts": [
        {**c, "dateOfLoss": c["dateOfLoss"][:10]} for c in d["NfipClaims"]]})


if __name__ == "__main__":
    main()
