META = dict(
    name="Epoch AI — Data on AI Models",
    domain="AI & Machine Learning",
    docs="https://epoch.ai/data/ai-models",
    auth="None",
    signup="No registration. CSV downloads updated continuously; licensed CC-BY (cite Epoch AI).",
    provides=["3,000+ AI models: organization, release date, domain, task, parameters, training compute (FLOP)",
              "Training dataset size, hardware, cost estimates, open-weights status, country",
              "Separate 'notable' and 'frontier' subsets"],
    ideas="Scaling-trend analysis (compute over time), forecasting training costs, open vs closed model studies.",
    endpoint="GET https://epoch.ai/data/{file}.csv",
    params=[
        ("all_ai_models.csv", "Every tracked model", "all_ai_models.csv"),
        ("notable_ai_models.csv", "Models meeting notability criteria (state of the art, highly cited, ...)", "notable_ai_models.csv"),
        ("frontier_ai_models.csv", "Models at the frontier of training compute when released", "frontier_ai_models.csv"),
        ("Columns", "Model, Organization, Publication date, Domain, Parameters, Training compute (FLOP), Open model weights?, ...", "—"),
    ],
)
import csv
import io

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    rows = list(csv.DictReader(io.StringIO(get("https://epoch.ai/data/notable_ai_models.csv", as_json=False))))
    with_compute = [r for r in rows if r["Training compute (FLOP)"]]
    largest = sorted(with_compute, key=lambda r: -float(r["Training compute (FLOP)"]))[:5]
    show({"notable_models": len(rows), "largest_training_compute": [
        {"model": r["Model"], "org": r["Organization"], "date": r["Publication date"],
         "training_FLOP": f'{float(r["Training compute (FLOP)"]):.1e}', "parameters": r["Parameters"] or None}
        for r in largest]})


if __name__ == "__main__":
    main()
