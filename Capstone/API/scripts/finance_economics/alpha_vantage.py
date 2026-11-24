META = dict(
    name="Alpha Vantage",
    domain="Finance & Economics",
    docs="https://www.alphavantage.co/documentation/",
    auth="API key (query `apikey`); `demo` works for documented examples (IBM)",
    env="ALPHAVANTAGE_API_KEY",
    signup="Claim a free key instantly at https://www.alphavantage.co/support/#api-key. Free: 25 req/day.",
    provides=["Daily/intraday OHLCV for stocks, ETFs, FX, crypto", "Fundamentals: income statement, balance sheet, earnings",
              "50+ technical indicators", "News & sentiment scores per ticker", "Economic indicators (CPI, GDP)"],
    ideas="Stock forecasting, news-sentiment trading signals, portfolio analysis.",
    endpoint="GET https://www.alphavantage.co/query",
    params=[
        ("function", "TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, GLOBAL_QUOTE, OVERVIEW, INCOME_STATEMENT, NEWS_SENTIMENT, SMA, RSI, CPI, ...", "TIME_SERIES_DAILY"),
        ("symbol", "Ticker", "IBM"),
        ("interval", "Intraday: 1min | 5min | 15min | 30min | 60min", "5min"),
        ("outputsize", "compact (latest 100 points) | full", "compact"),
        ("datatype", "json | csv", "json"),
        ("tickers / topics", "NEWS_SENTIMENT filters", "AAPL / technology"),
        ("apikey", "Your key (demo works for documented IBM examples)", "demo"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    import os
    d = get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_DAILY", "symbol": "IBM", "apikey": os.getenv("ALPHAVANTAGE_API_KEY", "demo")})
    ts = d["Time Series (Daily)"]
    show({"symbol": d["Meta Data"]["2. Symbol"],
          "last_5_days": [{"date": day, **{k[3:]: v for k, v in ts[day].items()}} for day in list(ts)[:5]]})


if __name__ == "__main__":
    main()
