META = dict(
    name="PubChem PUG-REST",
    domain="Science & Research",
    docs="https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
    auth="None",
    signup="No registration. Max 5 req/sec.",
    provides=["100M+ chemical compounds: formula, weight, SMILES, InChI, XLogP, synonyms",
              "Bioassay results, similarity/substructure search"],
    ideas="Molecular property prediction (cheminformatics + ML), drug-likeness models.",
    endpoint="GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{identifiers}/{operation}/{output}",
    params=[
        ("namespace", "cid | name | smiles | inchikey | formula", "cid"),
        ("identifiers", "Comma list of CIDs, or one name/SMILES", "2519,2244,3672"),
        ("operation", "property/<list> | synonyms | cids | description | record", "property/MolecularWeight,XLogP"),
        ("output", "JSON | CSV | XML | TXT | PNG", "JSON"),
        ("name → CID", "Look up a compound ID by name", "/compound/name/caffeine/cids/JSON"),
        ("fastsimilarity_2d", "Similar structures; Threshold= 0-100", "/compound/fastsimilarity_2d/cid/2244/cids/JSON?Threshold=95"),
    ],
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
