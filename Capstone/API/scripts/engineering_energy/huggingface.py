META = dict(
    name="Hugging Face Hub API",
    domain="Engineering, Energy & Tech",
    docs="https://huggingface.co/docs/hub/api",
    auth="None for public metadata (token for gated models / inference)",
    env="HF_TOKEN (optional)",
    signup="Optional: https://huggingface.co/settings/tokens -> New token (read). `pip install huggingface_hub`.",
    provides=["Model / dataset / Space search: downloads, likes, tags, pipeline task, license",
              "Model cards & dataset cards", "Datasets Server: row previews and Parquet files for any public dataset"],
    ideas="ML-ecosystem analytics, finding pretrained models & datasets for your capstone.",
    endpoint="GET https://huggingface.co/api/models",
    params=[
        ("search", "Substring of the model ID", "bert"),
        ("author", "Organization or user", "google"),
        ("pipeline_tag", "Task: text-classification, text-generation, image-classification, ...", "text-classification"),
        ("filter", "Tag filter (library, license, language, dataset)", "pytorch"),
        ("sort / direction", "downloads | likes | lastModified | trendingScore; -1 = descending", "downloads / -1"),
        ("limit", "Max results", "5"),
        ("full", "true → more metadata (files, card data)", "true"),
        ("/api/datasets?search=", "Sibling endpoint: dataset search (same params)", "search=imdb"),
        ("datasets-server.huggingface.co/rows?dataset=&config=&split=&offset=&length=", "Read rows of any public dataset (length ≤ 100)", "dataset=stanfordnlp/imdb&config=plain_text&split=train"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://huggingface.co/api/models", params={"pipeline_tag": "text-classification",
                                                         "sort": "downloads", "direction": -1, "limit": 5})
    show([{k: m.get(k) for k in ("id", "downloads", "likes", "library_name")} for m in d])


if __name__ == "__main__":
    main()
