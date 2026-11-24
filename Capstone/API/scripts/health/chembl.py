META = dict(
    name="ChEMBL (drug discovery bioactivity)",
    domain="Health & Social Science",
    docs="https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services",
    auth="None",
    signup="No registration. Python client: `pip install chembl_webresource_client`.",
    provides=["2.4M+ bioactive molecules with structures and properties (MW, logP, Lipinski, QED)",
              "20M+ bioactivity measurements (IC50, Ki, ...) against protein targets", "Approved drugs, mechanisms of action, indications"],
    ideas="QSAR / molecular property prediction, drug repurposing, target-ligand interaction models.",
    endpoint="GET https://www.ebi.ac.uk/chembl/api/data/{resource}.json",
    params=[
        ("molecule/{chembl_id}", "One molecule", "molecule/CHEMBL25 (aspirin)"),
        ("molecule?pref_name__iexact= / max_phase=", "Filter molecules (Django-style lookups: __iexact, __icontains, __gte)", "max_phase=4"),
        ("activity?target_chembl_id=&standard_type=", "Bioactivity measurements", "target_chembl_id=CHEMBL203&standard_type=IC50"),
        ("mechanism?molecule_chembl_id=", "Mechanism of action (often stored on the salt form, e.g. CHEMBL1703 not CHEMBL1431)", "molecule_chembl_id=CHEMBL1703"),
        ("drug_indication?molecule_chembl_id=", "Indications (MeSH/EFO)", "—"),
        ("limit / offset", "Paging (max 1,000)", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://www.ebi.ac.uk/chembl/api/data"


def main():
    # Mechanisms are recorded on the approved salt form (CHEMBL1703 = metformin hydrochloride), not the parent CHEMBL1431
    mol = get(f"{BASE}/molecule/CHEMBL1703.json")
    mech = get(f"{BASE}/mechanism.json", params={"molecule_chembl_id": "CHEMBL1703"})["mechanisms"]
    props = mol["molecule_properties"]
    show({"molecule": mol["pref_name"], "max_phase": mol["max_phase"],
          "properties": {k: props[k] for k in ("full_mwt", "alogp", "hba", "hbd", "num_ro5_violations", "qed_weighted")},
          "mechanisms": [{"action": m["action_type"], "mechanism": m["mechanism_of_action"]} for m in mech]})


if __name__ == "__main__":
    main()
