"""Price and split loaders (Yahoo Finance files from scripts/fetch_prices.py).

Yahoo's `Close` is split-adjusted, but the Dow is built from the REAL price on each day.
We rebuild it: raw_close = Close * (product of every split ratio dated after that day).
data/README.md shows this reproduces the Dow divisor exactly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICES = Path("data/raw/prices")
SPLITS = Path("data/raw/splits.csv")


def load_splits() -> pd.DataFrame:
    df = pd.read_csv(SPLITS, parse_dates=["date"])
    return df[["date", "ticker", "ratio"]].sort_values(["ticker", "date"]).reset_index(drop=True)


def load_prices(tickers: list[str] | None = None) -> pd.DataFrame:
    """All available price files (or just `tickers`) in the standard long format."""
    if tickers is None:
        tickers = [p.stem for p in PRICES.glob("*.csv") if not p.stem.startswith("_")]
    splits = load_splits()
    frames = []
    for t in sorted(tickers):
        path = PRICES / f"{t}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["Date"])
        df = pd.DataFrame({
            "date": raw["Date"],
            "ticker": t,
            "open": raw["Open"],
            "high": raw["High"],
            "low": raw["Low"],
            "close": raw["Close"],
            "adj_close": raw["Adj Close"],
            "volume": raw["Volume"],
        })
        factor = pd.Series(1.0, index=df.index)
        for d, r in splits.loc[splits.ticker == t, ["date", "ratio"]].itertuples(index=False):
            factor[df["date"] < d] *= r
        df["raw_close"] = df["close"] * factor
        frames.append(df.dropna(subset=["close"]))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
