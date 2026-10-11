META = dict(
    name="ClinicalTrials.gov API v2",
    domain="Health & Social Science",
    docs="https://clinicaltrials.gov/data-api/api",
    auth="None",
    signup="No registration (~50 req/min).",
    provides=["500k+ registered studies: conditions, interventions, phase, status, sponsors, enrollment, locations",
              "Results & adverse events for completed trials"],
    ideas="Trial-success prediction, geographic access to trials, sponsor landscape analysis.",
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
