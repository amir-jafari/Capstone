META = dict(
    name="RCSB Protein Data Bank (search + data APIs)",
    domain="Science & Research",
    docs="https://search.rcsb.org/ · https://data.rcsb.org/",
    auth="None",
    signup="No registration. Two APIs: Search (POST a JSON query, get IDs) and Data (GET details by ID, or GraphQL).",
    provides=["230k+ experimentally determined 3D structures of proteins, DNA, RNA and complexes",
              "Method (X-ray, cryo-EM, NMR), resolution, release date, organism, ligands, citations",
              "Full-text, sequence, structure-similarity and chemical searches"],
    ideas="Structural-biology datasets for ML (with AlphaFold), cryo-EM adoption trends, drug-target structure lookup.",
    endpoint="POST https://search.rcsb.org/rcsbsearch/v2/query  ·  GET https://data.rcsb.org/rest/v1/core/entry/{id}",
    params=[
        ("query (search body)", "{type: terminal, service: full_text | text | sequence | structure, parameters}", "{\"service\": \"full_text\", \"parameters\": {\"value\": \"spike glycoprotein\"}}"),
        ("return_type", "entry | polymer_entity | assembly | non_polymer_entity", "entry"),
        ("request_options.paginate", "{start, rows}", "{\"start\": 0, \"rows\": 5}"),
        ("request_options.sort", "[{sort_by, direction}], e.g. rcsb_accession_info.initial_release_date", "desc"),
        ("/rest/v1/core/entry/{id}", "Entry details: method, resolution, title, dates", "4HHB"),
        ("https://data.rcsb.org/graphql", "GraphQL alternative for batch details", "—"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, get, show


def main():
    q = {"query": {"type": "terminal", "service": "full_text", "parameters": {"value": "spike glycoprotein"}},
         "return_type": "entry",
         "request_options": {"paginate": {"start": 0, "rows": 4},
                             "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]}}
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    hits = r.json()
    out = []
    for h in hits["result_set"]:
        e = get(f'https://data.rcsb.org/rest/v1/core/entry/{h["identifier"]}')
        out.append({"pdb_id": h["identifier"], "title": e["struct"]["title"],
                    "method": e["exptl"][0]["method"], "resolution_A": (e.get("rcsb_entry_info") or {}).get("resolution_combined"),
                    "released": e["rcsb_accession_info"]["initial_release_date"][:10]})
    show({"matching_entries": hits["total_count"], "latest": out})


if __name__ == "__main__":
    main()
