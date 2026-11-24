META = dict(
    name="UCI Machine Learning Repository",
    domain="AI & Machine Learning",
    docs="https://github.com/uci-ml-repo/ucimlrepo",
    auth="None",
    signup="No registration. Python package: `pip install ucimlrepo` (`fetch_ucirepo(id=53)` returns pandas DataFrames).",
    provides=["600+ classic ML datasets: abstract, task type, number of instances/features, target column, DOI, citation",
              "Direct CSV download URL for each dataset", "Variable-level metadata (type, role, units, missing values)"],
    ideas="Classic benchmarks for teaching and baselines; fairness studies on Adult/German Credit.",
    endpoint="GET https://archive.ics.uci.edu/api/dataset",
    params=[
        ("id", "Dataset ID (from the dataset URL)", "53 (Iris)"),
        ("name", "Dataset name instead of id", "Iris"),
        ("data.data_url", "Field in the response: direct CSV download", "https://archive.ics.uci.edu/static/public/53/data.csv"),
        ("data.variables", "Field in the response: name, role (Feature/Target), type, units", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    out = []
    for i in (2, 186, 53):  # Adult, Wine Quality, Iris
        d = get("https://archive.ics.uci.edu/api/dataset", params={"id": i})["data"]
        out.append({k: d.get(k) for k in ("uci_id", "name", "area", "tasks", "num_instances", "num_features",
                                          "target_col", "data_url")})
    show(out)


if __name__ == "__main__":
    main()
