META = dict(
    name="The Met Museum Collection API",
    domain="Culture, Books & Entertainment",
    docs="https://metmuseum.github.io/",
    auth="None",
    signup="No registration. Max 80 req/sec. Get every object ID via `/public/collection/v1/objects`.",
    provides=["470k+ artworks: title, artist, date, culture, medium, department, dimensions, public-domain image URLs"],
    ideas="Computer-vision on art images, cultural-history analytics.",
    endpoint="GET https://collectionapi.metmuseum.org/public/collection/v1/objects/{objectID}",
    params=[
        ("objectID", "Object to fetch", "436524"),
        ("/objects", "All object IDs; filter with departmentIds= and metadataDate= (changed since)", "departmentIds=11"),
        ("/departments", "Department IDs and names", "—"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    # The /search endpoint was retired (HTTP 410); /objects (all IDs) and /objects/<id> still work.
    ids = [436524, 437984, 45734]  # Met object IDs (436524 = Van Gogh, 'Sunflowers')
    objs = [get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{i}") for i in ids]
    show([{k: o.get(k) for k in ("title", "artistDisplayName", "objectDate", "medium", "department", "primaryImageSmall")}
          for o in objs])


if __name__ == "__main__":
    main()
