META = dict(
    name="Hipolabs Universities List",
    domain="Education",
    docs="https://github.com/Hipo/university-domains-list",
    auth="None",
    signup="No registration. Community-maintained list (~10k universities); download the JSON from GitHub for bulk use.",
    provides=["University name, country, state/province, website, and email domains worldwide"],
    ideas="Mapping .edu email domains to institutions, linking institutions across datasets.",
    endpoint="GET http://universities.hipolabs.com/search",
    params=[
        ("name", "Substring of the university name", "George Washington"),
        ("country", "Country name", "United States"),
        ("domain", "Web domain", "gwu.edu"),
        ("limit / offset", "Paging", "5"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("http://universities.hipolabs.com/search", params={"name": "Washington", "country": "United States", "limit": 5})
    show([{"name": u["name"], "domains": u["domains"], "web": u["web_pages"][0]} for u in d])


if __name__ == "__main__":
    main()
