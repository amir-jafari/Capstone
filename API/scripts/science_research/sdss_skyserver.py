META = dict(
    name="SDSS SkyServer (Sloan Digital Sky Survey)",
    domain="Science & Research",
    docs="https://skyserver.sdss.org/dr18/en/help/docs/api.aspx",
    auth="None",
    signup="No registration. Run SQL against the SDSS catalog (hundreds of millions of objects). Queries time out after "
           "~10 minutes and return at most 500,000 rows; use CasJobs (free account) for larger jobs.",
    provides=["Photometric objects (PhotoObj): positions, magnitudes in u,g,r,i,z bands, object type (star/galaxy)",
              "Spectra (SpecObj): redshift, spectral class (GALAXY, QSO, STAR), velocity dispersion",
              "Image cutouts by coordinates (ImgCutout service)"],
    ideas="Galaxy/star/quasar classification from colors (classic ML task), photometric redshift regression.",
    endpoint="GET https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
    params=[
        ("cmd", "SQL (SQL Server dialect; use TOP n)", "select top 10 z, class from SpecObj where zWarning = 0"),
        ("format", "json | csv | xml | html", "json"),
        ("tables", "PhotoObj, SpecObj, Galaxy, Star, PhotoObjAll, SpecPhoto (joined view)", "SpecObj"),
        ("ImgCutout/getjpeg?ra=&dec=&scale=&width=&height=", "Sibling endpoint: JPEG cutout of the sky", "ra=180&dec=0&scale=0.4"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

SQL = """select class, count(*) as n, avg(z) as mean_redshift, max(z) as max_redshift
from SpecObj where zWarning = 0 group by class"""  # TOP n without ORDER BY returns a biased (nearby-galaxy) slice


def main():
    d = get("https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch", params={"cmd": SQL, "format": "json"})
    show({"sample": "all SDSS spectra with good redshifts (zWarning = 0)", "by_class": [
        {"class": r["class"], "count": r["n"], "mean_z": round(r["mean_redshift"], 3), "max_z": round(r["max_redshift"], 2)}
        for r in d[0]["Rows"]]})


if __name__ == "__main__":
    main()
