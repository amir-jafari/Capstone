META = dict(
    name="Massive (formerly Polygon.io) stock market API",
    domain="Finance & Economics",
    docs="https://massive.com/docs",
    auth="API key (query `apiKey` or Bearer header)",
    env="MASSIVE_API_KEY",
    signup="Polygon.io rebranded as Massive in 2025 (api.polygon.io still works). Sign up at https://massive.com for a "
           "free Basic key: 5 calls/minute, end-of-day data, 2 years of history.",
    provides=["US stocks, options, indices, forex, crypto: OHLCV bars (minute to monthly), previous close, grouped daily bars",
              "Ticker reference data, corporate actions (splits, dividends), financials, market status", "News with ticker tags"],
    ideas="Event studies (earnings, news), volatility modeling, building a clean daily price panel for many tickers.",
    endpoint="GET https://api.massive.com/{v2/aggs | v3/reference | v2/reference/news}/...",
    params=[
        ("v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{from}/{to}", "Bars; timespan minute | hour | day | week | month", "AAPL/range/1/day/2026-09-01/2026-09-30"),
        ("adjusted / sort / limit", "Split-adjusted; asc | desc; max 50,000", "true / asc / 120"),
        ("v2/aggs/grouped/locale/us/market/stocks/{date}", "Every US stock's bar for one day", "2026-10-09"),
        ("v3/reference/tickers?search=&active=", "Ticker search", "search=apple"),
        ("v3/reference/dividends?ticker=", "Dividends", "ticker=AAPL"),
        ("apiKey", "Your key", "<MASSIVE_API_KEY>"),
    ],
    sample="""{
  "ticker": "AAPL",
  "days": 21,
  "first_close": 228.4,
  "last_close": 236.1,
  "return_pct": 3.4,
  "avg_volume": 51234567
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    d = get("https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2026-09-01/2026-09-30",
            params={"adjusted": "true", "sort": "asc", "apiKey": key("MASSIVE_API_KEY")})
    bars = d["results"]
    show({"ticker": d["ticker"], "days": len(bars), "first_close": bars[0]["c"], "last_close": bars[-1]["c"],
          "return_pct": round(100 * (bars[-1]["c"] / bars[0]["c"] - 1), 1),
          "avg_volume": round(sum(b["v"] for b in bars) / len(bars))})


if __name__ == "__main__":
    main()
