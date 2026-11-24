META = dict(
    name="Wikidata Query Service (SPARQL)",
    domain="Culture, Books & Entertainment",
    docs="https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service",
    auth="None (descriptive User-Agent required)",
    signup="No registration. 60-second query timeout, one query at a time. Build and test queries interactively at "
           "https://query.wikidata.org first. CC0 data.",
    provides=["Structured facts on 110M+ items: films, awards, books, people, places, companies, artworks",
              "Arbitrary joins: e.g. award winners with release dates, directors, box office, birthplaces",
              "IDs that link to IMDb, MusicBrainz, Discogs, VIAF, OpenAlex and more"],
    ideas="Linking entertainment datasets via shared IDs, award-winner analyses, gender/diversity studies in film and music.",
    endpoint="GET https://query.wikidata.org/sparql",
    params=[
        ("query", "SPARQL query (wd: items, wdt: direct properties, p:/ps:/pq: for qualifiers)", "SELECT ?film WHERE { ?film wdt:P31 wd:Q11424 } LIMIT 5"),
        ("format", "json | xml | csv (or Accept header)", "json"),
        ("common IDs", "Q11424 film, P166 award received, P585 point in time, P57 director, P345 IMDb ID", "—"),
        ("SERVICE wikibase:label", "Adds human-readable ?xLabel variables", "bd:serviceParam wikibase:language \"en\""),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

QUERY = """
SELECT ?filmLabel ?year ?directorLabel WHERE {
  ?film wdt:P31 wd:Q11424 ;                      # instance of: film
        p:P166 ?award .
  ?award ps:P166 wd:Q102427 ; pq:P585 ?date .    # Academy Award for Best Picture, with date
  OPTIONAL { ?film wdt:P57 ?director . }
  BIND(YEAR(?date) AS ?year)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} ORDER BY DESC(?year) LIMIT 8
"""


def main():
    rows = get("https://query.wikidata.org/sparql", params={"query": QUERY, "format": "json"})["results"]["bindings"]
    show({"best_picture_winners": [{"year": r["year"]["value"], "film": r["filmLabel"]["value"],
                                    "director": r.get("directorLabel", {}).get("value")} for r in rows]}, max_items=8)


if __name__ == "__main__":
    main()
