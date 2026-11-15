"""Download macro series from FRED (no API key needed; uses the public CSV endpoint).

IMPORTANT for leakage: monthly series (CPI, unemployment) are dated by the month they
describe, but are published about two weeks after that month ends. Loaders must shift
them to their release date (we use observation month + 1 month + 15 days) before any
model sees them. Daily market series (rates, VIX, oil) are known at that day's close.

Output: data/raw/macro/<SERIES>.csv (DATE, value) and data/raw/macro/_summary.csv

Run:  uv run python scripts/fetch_macro.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/raw/macro")
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "DFF": ("Effective federal funds rate", "daily"),
    "DGS10": ("10-year Treasury yield", "daily"),
    "DGS2": ("2-year Treasury yield", "daily"),
    "T10Y2Y": ("10-year minus 2-year Treasury spread", "daily"),
    "T10Y3M": ("10-year minus 3-month Treasury spread", "daily"),
    "VIXCLS": ("CBOE VIX", "daily"),
    "DCOILWTICO": ("WTI crude oil price", "daily"),
    "DTWEXBGS": ("Broad US dollar index (from 2006)", "daily"),
    "BAMLH0A0HYM2": ("High-yield bond spread (ICE BofA)", "daily"),
    "CPIAUCSL": ("Consumer price index, all urban", "monthly, ~2-week release lag"),
    "UNRATE": ("Unemployment rate", "monthly, ~1-week release lag"),
}


def fetch(series: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            r = requests.get(URL, params={"id": series}, timeout=60,
                             headers={"User-Agent": "veritas-council-hackathon/0.1"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), na_values=".")
            df.columns = ["DATE", "value"]
            return df.dropna()
        except Exception as e:
            print(f"  {series}: attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, (name, freq) in SERIES.items():
        df = fetch(sid)
        if df.empty:
            print(f"MISSING {sid}")
            rows.append([sid, name, freq, "", "", 0, "missing"])
            continue
        df.to_csv(OUT / f"{sid}.csv", index=False)
        rows.append([sid, name, freq, df.DATE.iloc[0], df.DATE.iloc[-1], len(df), "ok"])
        print(f"ok {sid:13s} {df.DATE.iloc[0]} -> {df.DATE.iloc[-1]} ({len(df)} rows)")
    pd.DataFrame(rows, columns=["series", "name", "frequency", "first", "last", "rows", "status"]).to_csv(
        OUT / "_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
