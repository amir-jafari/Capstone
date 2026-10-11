META = dict(
    name="PubChem PUG-REST",
    domain="Science & Research",
    docs="https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
    auth="None",
    signup="No registration. Max 5 req/sec.",
    provides=["100M+ chemical compounds: formula, weight, SMILES, InChI, XLogP, synonyms",
              "Bioassay results, similarity/substructure search"],
    ideas="Molecular property prediction (cheminformatics + ML), drug-likeness models.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    # CIDs: 2519 caffeine, 2244 aspirin, 3672 ibuprofen (look up a CID via /compound/name/<name>/cids/JSON)
    props = "MolecularFormula,MolecularWeight,ConnectivitySMILES,XLogP,HBondDonorCount"
    d = get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2519,2244,3672/property/{props}/JSON")
    show(d["PropertyTable"]["Properties"])


if __name__ == "__main__":
    main()
