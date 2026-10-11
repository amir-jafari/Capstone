META = dict(
    name="Stack Exchange API",
    domain="Social Media",
    docs="https://api.stackexchange.com/docs",
    auth="None (optional key raises quota)",
    signup="Works without a key (300 req/day per IP). Register at https://stackapps.com/apps/oauth/register "
           "for a key -> 10,000 req/day.",
    provides=["Questions, answers, comments across 170+ sites (Stack Overflow, Cross Validated, ...)",
              "Score, views, tags, accepted answer, timestamps", "Tag popularity, user reputation"],
    ideas="Technology adoption trends, answer-quality prediction, developer-community analysis.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    items = get("https://api.stackexchange.com/2.3/questions", params={
        "order": "desc", "sort": "votes", "tagged": "pandas", "site": "stackoverflow", "pagesize": 5,
    })["items"]
    show([{"title": q["title"], "score": q["score"], "views": q["view_count"], "tags": q["tags"]} for q in items])


if __name__ == "__main__":
    main()
