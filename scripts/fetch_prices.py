"""Download daily prices, dividends and splits from Yahoo Finance (via yfinance).

Gets the Dow index (^DJI), the tradable Dow fund (DIA), every ticker in
data/raw/membership/dow_membership.csv, and the cross-asset series used by the
ML Forecaster (VIX, 10-year yield, dollar index, oil).

Yahoo's `Close` is split-adjusted, not dividend-adjusted. `Adj Close` is also
dividend-adjusted. The Dow uses REAL prices, so loaders rebuild them with
raw_close = Close * (product of split ratios dated after that day) (PLAN.md §0b).

Output:
  data/raw/prices/<TICKER>.csv   Date, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits
  data/raw/splits.csv            ticker, date, ratio
  data/raw/prices/_summary.csv   ticker, role, first, last, days, status

Run:  uv run python scripts/fetch_prices.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW = Path("data/raw")
PRICES = RAW / "prices"
START = "1990-01-01"

EXTRA = {
    "^DJI": "index",
    "DIA": "fund",
    "^VIX": "cross_asset",
    "^TNX": "cross_asset",
    "DX-Y.NYB": "cross_asset",
    "CL=F": "cross_asset",
}


def tickers() -> dict[str, str]:
    out = dict(EXTRA)
    with (RAW / "membership" / "dow_membership.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["yf_ticker"]:
                out.setdefault(row["yf_ticker"], "member")
    return out


def fetch(ticker: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=True)
            if not df.empty:
                df.index = df.index.tz_localize(None).normalize()
                df.index.name = "Date"
                return df
        except Exception as e:  # network hiccups: retry
            print(f"  {ticker}: attempt {attempt + 1} failed: {e}")
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    summary, splits = [], []
    for ticker, role in tickers().items():
        df = fetch(ticker)
        if df.empty:
            print(f"MISSING {ticker}")
            summary.append([ticker, role, "", "", 0, "missing"])
            continue
        df.to_csv(PRICES / f"{ticker}.csv")
        s = df.loc[df["Stock Splits"] > 0, "Stock Splits"]
        splits += [[ticker, d.date().isoformat(), r] for d, r in s.items()]
        summary.append([ticker, role, df.index[0].date(), df.index[-1].date(), len(df), "ok"])
        print(f"ok {ticker:9s} {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days, {len(s)} splits)")
    pd.DataFrame(summary, columns=["ticker", "role", "first", "last", "days", "status"]).to_csv(
        PRICES / "_summary.csv", index=False
    )
    pd.DataFrame(splits, columns=["ticker", "date", "ratio"]).to_csv(RAW / "splits.csv", index=False)


if __name__ == "__main__":
    main()
