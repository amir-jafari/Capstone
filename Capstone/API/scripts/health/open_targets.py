META = dict(
    name="Open Targets Platform (GraphQL)",
    domain="Health & Social Science",
    docs="https://platform-docs.opentargets.org/data-access/graphql-api",
    auth="None",
    signup="No registration. GraphQL: POST a query, get exactly the fields you ask for. Try queries in the browser at "
           "https://api.platform.opentargets.org/api/v4/graphql/browser. Diseases use EFO/MONDO IDs.",
    provides=["Target-disease association scores combining genetics, drugs, literature, animal models",
              "Known drugs per disease/target with clinical phase", "Target tractability and safety; disease ontology"],
    ideas="Drug-target prioritization, knowledge-graph link prediction, genetic evidence for drug repurposing.",
    endpoint="POST https://api.platform.opentargets.org/api/v4/graphql",
    params=[
        ("search(queryString:, entityNames:)", "Find IDs for diseases, targets or drugs", "queryString: \"type 2 diabetes\", entityNames: [\"disease\"]"),
        ("disease(efoId:)", "A disease by ID (MONDO_/EFO_), with associatedTargets, knownDrugs, ...", "MONDO_0005148"),
        ("target(ensemblId:)", "A gene/protein target, with associatedDiseases, tractability, safety", "ENSG00000148795"),
        ("drug(chemblId:)", "A drug with mechanisms and indications", "CHEMBL1431"),
        ("page: {index, size}", "Paging inside list fields", "{index: 0, size: 5}"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

URL = "https://api.platform.opentargets.org/api/v4/graphql"


def gql(query):
    r = requests.post(URL, json={"query": query}, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def main():
    disease_id = gql('{ search(queryString: "type 2 diabetes", entityNames: ["disease"], page: {index: 0, size: 1})'
                     ' { hits { id } } }')["search"]["hits"][0]["id"]
    d = gql(f'{{ disease(efoId: "{disease_id}") {{ name associatedTargets(page: {{index: 0, size: 5}}) '
            f'{{ count rows {{ target {{ approvedSymbol approvedName }} score }} }} }} }}')["disease"]
    show({"disease": f'{d["name"]} ({disease_id})', "associated_targets": d["associatedTargets"]["count"],
          "top_targets": [{"symbol": r["target"]["approvedSymbol"], "name": r["target"]["approvedName"],
                           "score": round(r["score"], 3)} for r in d["associatedTargets"]["rows"]]})


if __name__ == "__main__":
    main()
