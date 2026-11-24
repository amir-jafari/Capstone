META = dict(
    name="Datamuse (word-finding API)",
    domain="Education",
    docs="https://www.datamuse.com/api/",
    auth="None",
    signup="No registration. Up to 100,000 requests/day.",
    provides=["Words by meaning (reverse dictionary), sound, spelling pattern, rhyme, or topic",
              "Word associations, frequency, syllable counts, parts of speech", "Autocomplete suggestions (/sug)"],
    ideas="Vocabulary and reading-level tools, query expansion for search, word-game or language-learning apps.",
    endpoint="GET https://api.datamuse.com/words",
    params=[
        ("ml", "Means like (reverse dictionary)", "a person who studies data"),
        ("sl / sp", "Sounds like / spelled like (wildcards * and ?)", "sp=stat*"),
        ("rel_trg / rel_syn / rel_rhy", "Triggers (associated words) / synonyms / rhymes", "rel_trg=regression"),
        ("topics", "Bias results toward a topic", "statistics"),
        ("md", "Metadata flags: d definitions, p parts of speech, s syllables, f frequency", "dpf"),
        ("max", "Max results (up to 1,000)", "5"),
        ("/sug?s=", "Sibling endpoint: autocomplete", "s=statis"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    reverse = get("https://api.datamuse.com/words", params={"ml": "a person who studies data", "md": "p", "max": 5})
    assoc = get("https://api.datamuse.com/words", params={"rel_trg": "regression", "max": 8})
    show({"means_like_'a person who studies data'": [w["word"] for w in reverse],
          "words_associated_with_'regression'": [w["word"] for w in assoc]}, max_items=8)


if __name__ == "__main__":
    main()
