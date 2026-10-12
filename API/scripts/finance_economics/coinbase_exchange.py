META = dict(
    name="Coinbase Exchange public market data",
    domain="Finance & Economics",
    docs="https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-all-known-trading-pairs",
    auth="None for public market data",
    signup="No registration for public endpoints (about 10 requests/second). Accessible from the US (unlike some "
           "global exchanges' APIs). WebSocket feed available for real-time trades.",
    provides=["Trading pairs, OHLCV candles (1 min to 1 day), 24-hour stats, ticker, order book, recent trades",
              "Exchange-level prices rather than aggregated averages (compare with CoinGecko)"],
    ideas="Crypto volatility and microstructure (order-book depth), cross-exchange arbitrage, intraday seasonality.",
    endpoint="GET https://api.exchange.coinbase.com/products/{product_id}/{candles | ticker | stats | book | trades}",
    params=[
        ("product_id", "Trading pair", "BTC-USD"),
        ("granularity (candles)", "Seconds: 60, 300, 900, 3600, 21600, 86400", "86400"),
        ("start / end (candles)", "ISO 8601; max 300 candles per request", "2026-09-01T00:00:00Z"),
        ("level (book)", "1 best bid/ask | 2 aggregated | 3 full", "2"),
        ("/products", "All trading pairs with status and increments", "—"),
        ("candle row", "[time, low, high, open, close, volume]", "—"),
    ],
)
import datetime as dt
import statistics

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

BASE = "https://api.exchange.coinbase.com"


def main():
    candles = sorted(get(f"{BASE}/products/BTC-USD/candles", params={"granularity": 86400}))[-30:]  # oldest -> newest
    closes = [c[4] for c in candles]
    rets = [b / a - 1 for a, b in zip(closes, closes[1:])]
    stats = get(f"{BASE}/products/ETH-USD/stats")
    show({"BTC-USD_last_30_days": {"from": dt.date.fromtimestamp(candles[0][0]).isoformat(),
                                   "to": dt.date.fromtimestamp(candles[-1][0]).isoformat(),
                                   "close_first": closes[0], "close_last": closes[-1],
                                   "daily_volatility_pct": round(100 * statistics.stdev(rets), 2)},
          "ETH-USD_24h": {k: stats[k] for k in ("open", "high", "low", "last", "volume")}})


if __name__ == "__main__":
    main()
