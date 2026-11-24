META = dict(
    name="UniProt (proteins)",
    domain="Health & Social Science",
    docs="https://www.uniprot.org/help/api",
    auth="None",
    signup="No registration. For large result sets use the /stream endpoint or cursor paging (Link header).",
    provides=["250M+ protein sequences; 570k+ expert-reviewed (Swiss-Prot) entries with function, disease involvement, variants",
              "Gene names, organism, subcellular location, structures (PDB/AlphaFold), GO terms, cross-references"],
    ideas="Protein-function prediction, disease-variant analysis, sequence datasets for ML (embeddings).",
    endpoint="GET https://rest.uniprot.org/uniprotkb/{search | stream | {accession}}",
    params=[
        ("query", "Query syntax: gene:, organism_id:, reviewed:true, disease:, keyword:", "gene:BRCA1 AND organism_id:9606 AND reviewed:true"),
        ("fields", "Columns to return", "accession,gene_names,protein_name,length,cc_disease"),
        ("format", "json | tsv | fasta | xml", "json"),
        ("size", "Page size (max 500); follow the Link header for more", "5"),
        ("/uniprotkb/{accession}", "One entry", "/uniprotkb/P38398"),
        ("/uniref, /uniparc, /proteomes", "Other UniProt databases with the same pattern", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://rest.uniprot.org/uniprotkb/search", params={
        "query": "gene:BRCA1 AND organism_id:9606 AND reviewed:true", "format": "json",
        "fields": "accession,gene_names,protein_name,length,cc_disease", "size": 1})["results"][0]
    show({"accession": d["primaryAccession"], "gene": d["genes"][0]["geneName"]["value"],
          "protein": d["proteinDescription"]["recommendedName"]["fullName"]["value"], "length_aa": d["sequence"]["length"],
          "diseases": [c["disease"]["diseaseId"] for c in d.get("comments", []) if c.get("disease")]})


if __name__ == "__main__":
    main()
