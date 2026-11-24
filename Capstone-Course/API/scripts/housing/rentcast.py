META = dict(
    name="RentCast (property records, rent & value estimates)",
    domain="Housing & Real Estate",
    docs="https://developers.rentcast.io/reference/introduction",
    auth="API key (header `X-Api-Key`)",
    env="RENTCAST_API_KEY",
    signup="Create an account at https://app.rentcast.io/app/api and activate the free Developer plan "
           "(50 requests/month; paid plans for more). Commercial terms apply - read them before publishing data.",
    provides=["Property records for 140M+ US properties: beds, baths, sq ft, year built, last sale, tax assessments",
              "Automated rent and value estimates (AVM) with comparable listings",
              "Active sale/rental listings and ZIP-level market statistics with history"],
    ideas="Rent-prediction models, landlord/investor yield analysis, small-area rent trends.",
    endpoint="GET https://api.rentcast.io/v1/{resource}",
    params=[
        ("markets?zipCode=&dataType=&historyRange=", "ZIP market stats; dataType All | Sale | Rental; history in months", "zipCode=20052&dataType=Rental"),
        ("avm/rent/long-term?address=&propertyType=&bedrooms=", "Rent estimate + comparables for an address", "address=2121 I St NW, Washington, DC 20052"),
        ("avm/value?address=", "Value estimate + comparables", "—"),
        ("properties?address= / ?zipCode=", "Property records", "zipCode=20052&limit=5"),
        ("listings/rental/long-term?city=&state=&status=", "Active rental listings", "city=Washington&state=DC&status=Active"),
        ("X-Api-Key (header)", "Your key", "<RENTCAST_API_KEY>"),
    ],
    sample="""{
  "zipCode": "20052",
  "rental": {"averageRent": 2850, "medianRent": 2600, "minRent": 1450, "maxRent": 7900, "totalListings": 214},
  "sale": {"averagePrice": 612000, "medianPrice": 489000, "totalListings": 38}
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.rentcast.io/v1/markets", headers={"X-Api-Key": key("RENTCAST_API_KEY")},
            params={"zipCode": "20052", "dataType": "All", "historyRange": 6})
    rent, sale = d.get("rentalData", {}), d.get("saleData", {})
    show({"zipCode": d["zipCode"],
          "rental": {k: rent.get(k) for k in ("averageRent", "medianRent", "minRent", "maxRent", "totalListings")},
          "sale": {k: sale.get(k) for k in ("averagePrice", "medianPrice", "totalListings")}})


if __name__ == "__main__":
    main()
