META = dict(
    name="USDA NASS CropScape (Cropland Data Layer)",
    domain="Agriculture & Food",
    docs="https://nassgeodata.gmu.edu/CropScape/devhelp/help.html",
    auth="None",
    signup="No registration. The Cropland Data Layer is a 30 m satellite-derived map of what is planted on every field "
           "in the US each year (2008-present nationally).",
    provides=["Acreage by crop type for any county, state, ASD or custom area (from the 30 m crop map)",
              "Raster extracts (GeoTIFF) of the crop map for an area and year", "Pixel value at a point; change between years"],
    ideas="Crop rotation and land-use change, irrigation/water demand estimates, pollinator-habitat studies.",
    endpoint="GET https://nassgeodata.gmu.edu/axis2/services/CDLService/{GetCDLStat | GetCDLFile | GetCDLValue}",
    params=[
        ("year", "Crop year", "2023"),
        ("fips", "State (2-digit) or county (5-digit) FIPS", "19169 (Story County, Iowa)"),
        ("format", "json | csv (GetCDLStat) - returns an XML wrapper with a returnURL to the file", "json"),
        ("bbox / crs", "Custom area for GetCDLFile / GetCDLStat (Albers coordinates)", "—"),
        ("GetCDLValue?year=&x=&y=", "Crop at a point (Albers x/y)", "—"),
    ],
)
import re

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

ROW = re.compile(r'\{value:(\d+), count:(\d+), category:"([^"]+)", color:"[^"]*", acreage:([\d.]+)\}')


def main():
    wrapper = get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
                  params={"year": 2023, "fips": "19169", "format": "json"}, as_json=False)
    file_url = re.search(r"<returnURL>(.*?)</returnURL>", wrapper).group(1)
    text = get(file_url, as_json=False)  # JavaScript-style object (unquoted keys), so parse with a regex
    rows = sorted(((cat, float(acres)) for _, _, cat, acres in ROW.findall(text)), key=lambda r: -r[1])
    show({"county": "Story County, Iowa (2023)", "top_land_cover_acres": [{"category": c, "acres": a} for c, a in rows[:6]]})


if __name__ == "__main__":
    main()
