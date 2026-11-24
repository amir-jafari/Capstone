META = dict(
    name="Open Trivia Database",
    domain="Education",
    docs="https://opentdb.com/api_config.php",
    auth="None",
    signup="No registration. One request per 5 seconds per IP. CC BY-SA 4.0. Use a session token to avoid repeat questions.",
    provides=["4,000+ verified multiple-choice and true/false questions in 24 categories with difficulty levels",
              "Category list and per-category question counts"],
    ideas="Quiz/edtech app prototypes, evaluating LLMs on trivia by difficulty, question-difficulty prediction.",
    endpoint="GET https://opentdb.com/api.php",
    params=[
        ("amount", "Number of questions (max 50)", "5"),
        ("category", "Category ID (list via /api_category.php), e.g. 17 Science & Nature, 18 Computers, 19 Mathematics", "18"),
        ("difficulty", "easy | medium | hard", "medium"),
        ("type", "multiple | boolean", "multiple"),
        ("encode", "url3986 | base64 (default is HTML entities)", "url3986"),
        ("token", "Session token from /api_token.php?command=request", "—"),
    ],
)
import html

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    cats = get("https://opentdb.com/api_category.php")["trivia_categories"]
    d = get("https://opentdb.com/api.php", params={"amount": 3, "category": 18, "difficulty": "medium", "type": "multiple"})
    show({"categories": len(cats), "computer_science_questions": [
        {"question": html.unescape(q["question"]), "answer": html.unescape(q["correct_answer"]),
         "distractors": [html.unescape(a) for a in q["incorrect_answers"]]} for q in d["results"]]})


if __name__ == "__main__":
    main()
