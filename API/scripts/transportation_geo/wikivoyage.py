META = dict(
    name="Wikivoyage (MediaWiki API)",
    domain="Transportation & Geospatial",
    docs="https://www.mediawiki.org/wiki/API:Main_page",
    auth="None",
    signup="No registration. The same MediaWiki API works for Wikipedia and every Wikimedia wiki; send a descriptive "
           "User-Agent. CC BY-SA content.",
    provides=["Free travel guides for 30,000+ destinations: understand, get in, get around, see, do, eat, sleep, stay safe",
              "Plain-text extracts, section lists, coordinates, links between destinations, page views"],
    ideas="Destination recommendation with NLP, travel-guide summarization, mapping attractions by coordinates.",
    endpoint="GET https://en.wikivoyage.org/w/api.php",
    params=[
        ("action", "query | parse | opensearch", "query"),
        ("prop=extracts&explaintext=1", "Plain-text content (exintro=1 for the intro only)", "—"),
        ("titles", "Page title(s), | separated", "Washington, D.C."),
        ("action=parse&prop=sections", "Section list of a page (then parse&section=N for one section)", "page=Washington, D.C."),
        ("prop=coordinates", "Latitude/longitude of the destination", "—"),
        ("format", "json", "json"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

API = "https://en.wikivoyage.org/w/api.php"


def main():
    page = "Washington, D.C."
    intro = get(API, params={"action": "query", "prop": "extracts|coordinates", "exintro": 1, "explaintext": 1,
                             "titles": page, "format": "json"})["query"]["pages"]
    p = next(iter(intro.values()))
    sections = get(API, params={"action": "parse", "page": page, "prop": "sections", "format": "json"})["parse"]["sections"]
    show({"destination": p["title"], "coordinates": (p.get("coordinates") or [{}])[0],
          "intro": p["extract"], "top_sections": [s["line"] for s in sections if s["toclevel"] == 1]},
         max_items=14, max_str=300)


if __name__ == "__main__":
    main()
