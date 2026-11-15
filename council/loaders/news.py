"""News loaders.

Market news: the top 25 r/worldnews headlines per day, from two sources built the same way:
  - Kaggle Combined_News_DJIA.csv (2008-08 to 2016-07)
  - our Arctic Shift rebuild in reddit_days/*.json (2016-07 onward, still backfilling)
Company news: Kaggle "Massive Stock News" ticker-tagged headlines (2009-02 to 2020-06).

`date` is the calendar day the headline was published. Reddit days are UTC, so some
headlines appear after the US close; the Historian therefore only ever gives a model
news dated strictly BEFORE the decision date.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

NEWS = Path("data/raw/news")
_BYTES = re.compile(r"""^b(['"])(.*)\1$""", re.S)


def _clean(text: str) -> str:
    """Strip the b'...' wrappers the Kaggle file has on many headlines."""
    text = str(text).strip()
    m = _BYTES.match(text)
    return m.group(2).strip() if m else text


def load_market_news() -> pd.DataFrame:
    kaggle = pd.read_csv(NEWS / "stocknews" / "Combined_News_DJIA.csv", parse_dates=["Date"])
    long = kaggle.melt(id_vars=["Date"], value_vars=[f"Top{i}" for i in range(1, 26)],
                       var_name="rank", value_name="headline").dropna(subset=["headline"])
    k = pd.DataFrame({
        "date": long["Date"],
        "ticker": "DJIA",
        "headline": long["headline"].map(_clean),
        "source": "kaggle_worldnews",
        "score": float("nan"),
    })
    rows = []
    for path in sorted((NEWS / "reddit_days").glob("*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for post in day["top25"]:
            rows.append((pd.Timestamp(day["date"]), "DJIA", post["title"], "arcticshift_worldnews", post["score"]))
    r = pd.DataFrame(rows, columns=["date", "ticker", "headline", "source", "score"])
    r = r[r["date"] > k["date"].max()]  # the two sources never overlap
    out = pd.concat([k, r], ignore_index=True)
    return out[out["headline"].str.len() > 0].sort_values("date").reset_index(drop=True)


def load_company_news(tickers: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for name in ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]:
        df = pd.read_csv(NEWS / "company" / name, usecols=["headline", "publisher", "date", "stock"])
        if tickers is not None:
            df = df[df["stock"].isin(tickers)]
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(df["date"].str[:10], errors="coerce"),
            "ticker": df["stock"],
            "headline": df["headline"].astype(str),
            "source": df["publisher"].fillna("unknown"),
            "score": float("nan"),
        }))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    out = out[out["date"] >= "2009-01-01"]  # drops the 1969 placeholder dates
    return out.drop_duplicates(["date", "ticker", "headline"]).sort_values(["ticker", "date"]).reset_index(drop=True)
