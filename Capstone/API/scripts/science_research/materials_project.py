META = dict(
    name="Materials Project API",
    domain="Science & Research",
    docs="https://docs.materialsproject.org/downloading-data/using-the-api",
    auth="API key (header `X-API-KEY`)",
    env="MP_API_KEY",
    signup="Create a free account at https://next-gen.materialsproject.org and copy the key from your dashboard. "
           "Python client: `pip install mp-api` (MPRester).",
    provides=["150k+ inorganic compounds computed with DFT: formation energy, stability (energy above hull), band gap, density",
              "Crystal structures, elastic and magnetic properties, battery electrodes, phonons, X-ray diffraction",
              "Search by formula, elements, chemical system, or property ranges"],
    ideas="Materials-property prediction (band gap, stability) with ML, battery-material screening.",
    endpoint="GET https://api.materialsproject.org/materials/summary/",
    params=[
        ("formula / elements / chemsys", "Composition filters", "chemsys=Li-Fe-O"),
        ("band_gap_min / band_gap_max", "Property ranges (also energy_above_hull_max, density_min, ...)", "1 / 3"),
        ("is_stable", "true -> on the convex hull", "true"),
        ("_fields", "Fields to return", "material_id,formula_pretty,band_gap,energy_above_hull,density"),
        ("_limit / _skip", "Paging", "5"),
        ("X-API-KEY (header)", "Your key", "<MP_API_KEY>"),
    ],
    sample="""[
  {"material_id": "mp-19017", "formula_pretty": "LiFePO4", "band_gap": 3.70, "energy_above_hull": 0.0, "density": 3.52},
  {"material_id": "mp-18900", "formula_pretty": "LiFeO2", "band_gap": 1.84, "energy_above_hull": 0.0, "density": 4.31}
]""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.materialsproject.org/materials/summary/", headers={"X-API-KEY": key("MP_API_KEY")}, params={
        "chemsys": "Li-Fe-O,Li-Fe-P-O", "is_stable": "true", "_limit": 5,
        "_fields": "material_id,formula_pretty,band_gap,energy_above_hull,density"})
    show(d["data"])


if __name__ == "__main__":
    main()
