META = dict(
    name="ClinicalTrials.gov API v2",
    domain="Health & Social Science",
    docs="https://clinicaltrials.gov/data-api/api",
    auth="None",
    signup="No registration (~50 req/min).",
    provides=["500k+ registered studies: conditions, interventions, phase, status, sponsors, enrollment, locations",
              "Results & adverse events for completed trials"],
    ideas="Trial-success prediction, geographic access to trials, sponsor landscape analysis.",
    endpoint="GET https://clinicaltrials.gov/api/v2/studies",
    params=[
        ("query.cond / query.intr / query.term", "Condition / intervention / general search", "Alzheimer Disease"),
        ("query.locn / query.spons", "Location / sponsor search", "Washington, DC"),
        ("filter.overallStatus", "RECRUITING | COMPLETED | ACTIVE_NOT_RECRUITING | ... (comma list)", "RECRUITING"),
        ("filter.geo", "distance(lat,lon,radius)", "distance(38.9,-77.04,50mi)"),
        ("fields", "Fields to return (piece names)", "NCTId,BriefTitle,Phase"),
        ("pageSize / pageToken", "Max 1,000 per page; nextPageToken for paging", "5"),
        ("countTotal", "true → include totalCount", "true"),
        ("format", "json | csv", "json"),
        ("/api/v2/studies/{nctId}", "Sibling endpoint: one full study record", "NCT05123456"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://clinicaltrials.gov/api/v2/studies", params={
        "query.cond": "Alzheimer Disease", "filter.overallStatus": "RECRUITING", "pageSize": 5,
        "fields": "NCTId,BriefTitle,Phase,LeadSponsorName,EnrollmentCount", "countTotal": "true"})
    show({"total_recruiting": d.get("totalCount"), "studies": [
        {"nct": s["protocolSection"]["identificationModule"]["nctId"],
         "title": s["protocolSection"]["identificationModule"]["briefTitle"],
         "phase": s["protocolSection"].get("designModule", {}).get("phases"),
         "sponsor": s["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"]}
        for s in d["studies"]]})


if __name__ == "__main__":
    main()
