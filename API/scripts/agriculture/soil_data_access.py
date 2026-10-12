META = dict(
    name="USDA NRCS Soil Data Access (SSURGO)",
    domain="Agriculture & Food",
    docs="https://sdmdataaccess.nrcs.usda.gov/WebServiceHelp.aspx",
    auth="None",
    signup="No registration. Send SQL (T-SQL) queries against the national SSURGO soil database; max 100,000 rows or "
           "~32 MB per request.",
    provides=["Detailed soil survey for the US: map units, components, horizons",
              "Properties: texture, organic matter, pH, drainage class, available water capacity, farmland classification",
              "Spatial functions to query by point or polygon (WKT)"],
    ideas="Soil covariates for yield models, prime-farmland loss to development, land-suitability analysis.",
    endpoint="POST https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest",
    params=[
        ("query (JSON body)", "T-SQL against SSURGO tables: sacatalog, legend, mapunit, component, chorizon, ...", "SELECT TOP 5 areasymbol, areaname FROM sacatalog"),
        ("format (JSON body)", "JSON | JSON+COLUMNNAME | JSON+COLUMNNAME+METADATA | XML", "JSON+COLUMNNAME"),
        ("SDA_Get_Mukey_from_intersection_with_WktWgs84('point(lon lat)')", "Table function: map units at a point", "point(-93.6 42.0)"),
        ("farmlndcl (mapunit)", "Farmland classification, e.g. All areas are prime farmland", "—"),
    ],
)
import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show

SQL = """
SELECT TOP 5 mu.musym, mu.muname, mu.farmlndcl, c.compname, c.comppct_r, c.drainagecl
FROM mapunit mu
JOIN component c ON c.mukey = mu.mukey
WHERE mu.mukey IN (SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point(-93.6 42.0)'))
ORDER BY c.comppct_r DESC
"""


def main():
    r = requests.post("https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest", timeout=60,
                      headers={"User-Agent": USER_AGENT}, json={"query": SQL, "format": "JSON+COLUMNNAME"})
    r.raise_for_status()
    header, *rows = r.json()["Table"]
    show({"point": "-93.6, 42.0 (central Iowa)", "soil_components": [dict(zip(header, row)) for row in rows]})


if __name__ == "__main__":
    main()
