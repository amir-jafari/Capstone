META = dict(
    name="CERN Open Data Portal",
    domain="Science & Research",
    docs="https://opendata.cern.ch/docs/about",
    auth="None",
    signup="No registration. Real LHC collision data (CMS, ATLAS, LHCb, ALICE) plus simulations, software and "
           "educational datasets small enough for a laptop. Bulk files via HTTP or the cernopendata-client.",
    provides=["Record search over datasets, simulated data, software, documentation; experiment, year, type, size",
              "Derived/educational datasets (e.g. dimuon events for Higgs/Z-boson exercises) as CSV",
              "File lists with sizes and download URIs"],
    ideas="Particle-classification ML on real collision events, reproducing the Higgs/Z peak from the dimuon spectrum.",
    endpoint="GET https://opendata.cern.ch/api/records/",
    params=[
        ("q", "Search text", "higgs"),
        ("type", "Dataset | Software | Documentation; add a subtype as type=Dataset::Derived (Collision | Simulated | Derived)", "Dataset::Derived"),
        ("experiment", "CMS | ATLAS | LHCb | ALICE | ...", "CMS"),
        ("size / page / sort", "Paging and sort (mostrecent, bestmatch)", "5"),
        ("/api/records/{recid}", "One record with files", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    # Subtypes are given as type=Dataset::Derived (a separate subtype= parameter is ignored)
    d = get("https://opendata.cern.ch/api/records/", params={"q": "dimuon", "type": "Dataset::Derived", "size": 5})
    show({"matching_educational_datasets": d["hits"]["total"], "examples": [
        {"recid": h["metadata"].get("recid"), "title": h["metadata"]["title"],
         "experiment": h["metadata"].get("experiment"), "year": h["metadata"].get("date_created"),
         "files": len(h["metadata"].get("files", []))} for h in d["hits"]["hits"]]})


if __name__ == "__main__":
    main()
