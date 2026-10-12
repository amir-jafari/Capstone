META = dict(
    name="OpenML",
    domain="AI & Machine Learning",
    docs="https://www.openml.org/apis",
    auth="None for reading (free key to upload runs)",
    env="OPENML_API_KEY (optional)",
    signup="No registration to read. Create an account at https://www.openml.org for a key to upload results. "
           "Python package: `pip install openml`.",
    provides=["20,000+ ML datasets with meta-features (instances, features, classes, missing values)",
              "Tasks (dataset + target + evaluation split), benchmark suites (e.g. OpenML-CC18)",
              "Millions of runs: model + hyperparameters + evaluation scores"],
    ideas="Meta-learning (predict which model wins from dataset features), AutoML benchmarking, reproducible baselines.",
    endpoint="GET https://www.openml.org/api/v1/json/{resource}",
    params=[
        ("data/list/{filters}", "Dataset list; path filters limit/N, offset/N, status/active, tag/X, number_instances/A..B", "data/list/limit/5/number_instances/1000..5000"),
        ("data/{id}", "One dataset's description, licence, default target, file URL", "data/61 (iris)"),
        ("data/qualities/{id}", "Meta-features for a dataset", "data/qualities/61"),
        ("task/list/{filters}", "Tasks; filters type/1 (classification), tag/, data_id/", "task/list/type/1/limit/5"),
        ("evaluation/list/function/{metric}/task/{id}", "Best runs on a task by a metric", "function/predictive_accuracy/task/59"),
        ("study/{id}", "Benchmark suite contents", "study/99 (OpenML-CC18)"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://www.openml.org/api/v1/json/data/list/limit/200/status/active/number_instances/1000..5000")
    rows = []
    for ds in d["data"]["dataset"][:5]:
        q = {x["name"]: int(float(x["value"])) for x in ds.get("quality", [])}  # values arrive as "3196.0"
        rows.append({"id": ds["did"], "name": ds["name"], "instances": q.get("NumberOfInstances"),
                     "features": q.get("NumberOfFeatures"), "classes": q.get("NumberOfClasses"),
                     "missing_values": q.get("NumberOfMissingValues")})
    show(rows)


if __name__ == "__main__":
    main()
