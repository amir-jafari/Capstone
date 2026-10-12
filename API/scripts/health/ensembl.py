META = dict(
    name="Ensembl REST API (genomics)",
    domain="Health & Social Science",
    docs="https://rest.ensembl.org/",
    auth="None",
    signup="No registration. Max 15 requests/second (55,000/hour). Send content-type=application/json.",
    provides=["Genes, transcripts and their genomic coordinates for human and 300+ species",
              "Genetic variants with consequences (VEP), population allele frequencies, phenotype associations",
              "Sequences, homologues across species, regulatory features"],
    ideas="Variant-effect analysis pipelines, genomic feature engineering, cross-species comparison.",
    endpoint="GET https://rest.ensembl.org/{resource}",
    params=[
        ("lookup/symbol/{species}/{symbol}", "Gene by symbol (expand=1 adds transcripts)", "lookup/symbol/homo_sapiens/BRCA1"),
        ("variation/{species}/{rsid}", "A variant: alleles, location, consequence, clinical significance; pops=1 adds population frequencies", "variation/human/rs429358?pops=1"),
        ("vep/{species}/hgvs/{notation}", "Variant Effect Predictor", "vep/human/hgvs/ENST00000366667:c.803C>T"),
        ("sequence/id/{id}?type=", "Genomic, cDNA or protein sequence", "type=protein"),
        ("homology/symbol/{species}/{symbol}", "Orthologues in other species", "—"),
        ("content-type", "application/json (query parameter or Accept header)", "application/json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://rest.ensembl.org"
JSON = {"content-type": "application/json"}


def main():
    gene = get(f"{BASE}/lookup/symbol/homo_sapiens/APOE", params=JSON)
    var = get(f"{BASE}/variation/human/rs429358", params={**JSON, "pops": 1})  # pops=1 adds population frequencies
    show({"gene": {k: gene[k] for k in ("id", "display_name", "biotype", "seq_region_name", "start", "end", "description")},
          "variant": {"id": var["name"], "location": var["mappings"][0]["location"],
                      "alleles": var["mappings"][0]["allele_string"], "consequence": var.get("most_severe_consequence"),
                      "gnomAD_allele_freq": {p["allele"]: p["frequency"] for p in var.get("populations", [])
                                             if p["population"] == "gnomADg:ALL"},
                      "clinical_significance": var.get("clinical_significance")}})


if __name__ == "__main__":
    main()
