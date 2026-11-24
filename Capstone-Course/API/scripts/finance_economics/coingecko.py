META = dict(
    name="CoinGecko",
    domain="Finance & Economics",
    docs="https://docs.coingecko.com/v3.0.1/reference/introduction",
    auth="None for public endpoints (free 'Demo' key recommended)",
    signup="Works keyless at low volume. For a stable 30 calls/min, 10k/month register a free Demo key at "
           "https://www.coingecko.com/en/developers/dashboard and pass header `x-cg-demo-api-key`.",
    provides=["Prices, market cap, volume for 15k+ crypto assets", "Historical OHLC & market charts",
              "Exchanges, trending coins, categories"],
    ideas="Crypto volatility modeling, cross-asset correlation, social-hype vs price.",
    endpoint="GET https://api.coingecko.com/api/v3/coins/markets",
    params=[
        ("vs_currency", "Quote currency", "usd"),
        ("ids", "Comma-separated coin IDs (list via /coins/list)", "bitcoin,ethereum"),
        ("order", "market_cap_desc | volume_desc | id_asc ...", "market_cap_desc"),
        ("per_page / page", "Max 250 per page", "5"),
        ("price_change_percentage", "Extra change windows: 1h,24h,7d,14d,30d,200d,1y", "1h,7d"),
        ("/simple/price?ids=&vs_currencies=", "Sibling endpoint: quick spot prices", "ids=bitcoin&vs_currencies=usd"),
        ("/coins/{id}/market_chart?vs_currency=&days=", "Sibling endpoint: historical price/market cap/volume", "bitcoin, days=365"),
        ("x-cg-demo-api-key (header)", "Optional free Demo key", "<key>"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 5})
    show([{k: c[k] for k in ("name", "current_price", "market_cap", "price_change_percentage_24h")} for c in d])


if __name__ == "__main__":
    main()
