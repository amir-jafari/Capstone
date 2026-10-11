META = dict(
    name="arXiv API",
    domain="Science & Research",
    docs="https://info.arxiv.org/help/api/user-manual.html",
    auth="None",
    signup="No registration. Wait ~3 seconds between calls. Returns Atom XML.",
    provides=["Preprints in CS, stats, physics, math, econ, q-bio: title, abstract, authors, categories, dates, PDF link"],
    ideas="Abstract topic modeling, paper recommendation, tracking ML research trends.",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
import xml.etree.ElementTree as ET

from _common import get, show

NS = {"a": "http://www.w3.org/2005/Atom"}


def main():
    xml = get("http://export.arxiv.org/api/query", params={
        "search_query": "cat:cs.CL AND abs:retrieval", "sortBy": "submittedDate", "max_results": 3}, as_json=False)
    root = ET.fromstring(xml)
    show([{"title": " ".join(e.find("a:title", NS).text.split()),
           "published": e.find("a:published", NS).text,
           "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:3],
           "abstract": " ".join(e.find("a:summary", NS).text.split())}
          for e in root.findall("a:entry", NS)], max_str=200)


if __name__ == "__main__":
    main()
