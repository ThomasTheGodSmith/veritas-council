"""Realistic trades: what a purchase actually returns, day by day, with a stop-loss.

A purchase dated t is bought at the next day's open and sold at the close h trading days after t
(the same timing as the labels in council/features.py). With a stop-loss at stop_pct below the
entry price, each day of the holding period is checked in order:
  - if that day OPENS at or below the stop price (bad overnight news), it sells at the open,
    which can be well below the stop: this is the "gap" a real stop can't protect against;
  - otherwise, if that day's LOW reaches the stop price, it sells at the stop price;
  - if neither happens on any day, it sells at the final close.
All prices are dividend- and split-adjusted, so returns are what an investor really got.

Also: overnight_gap_q95(), the 1-in-20 bad overnight drop for a stock, measured on training years.
The Judge places the stop that much tighter than the user's max loss, so a 1-in-20 gap through
the stop still keeps the loss within the limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H = 20


def _adjusted(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    g = prices[prices.ticker == ticker].sort_values("date")
    f = g["adj_close"] / g["close"]
    return pd.DataFrame({"date": g["date"].values, "open": (g["open"] * f).values,
                         "low": (g["low"] * f).values, "close": g["adj_close"].values})


def simulate(rows: pd.DataFrame, prices: pd.DataFrame, amount: np.ndarray,
             stop_pct: np.ndarray | None, h: int = H) -> pd.DataFrame:
    """One result per row: profit (in the same money as `amount`), worst paper loss before the
    sale, whether the stop fired, and whether it fired on a gap. amount 0 = didn't buy."""
    amount = np.asarray(amount, float)
    stop = np.full(len(rows), np.inf) if stop_pct is None else np.asarray(stop_pct, float)
    profit = np.zeros(len(rows))
    worst = np.zeros(len(rows))
    stopped = np.zeros(len(rows), bool)
    gapped = np.zeros(len(rows), bool)
    rows = rows.reset_index(drop=True)
    for ticker, idx in rows.groupby("ticker").groups.items():
        px = _adjusted(prices, ticker)
        pos = pd.Series(np.arange(len(px)), index=px["date"])
        i = pos.reindex(rows.loc[idx, "date"]).to_numpy()
        ok = ~np.isnan(i) & (np.nan_to_num(i, nan=-1) + h < len(px))
        idx = np.asarray(idx)[ok]
        i = i[ok].astype(int)
        O, L, C = px["open"].to_numpy(), px["low"].to_numpy(), px["close"].to_numpy()
        days = i[:, None] + np.arange(1, h + 1)[None, :]            # t+1 .. t+h
        entry = O[i + 1]
        level = entry * (1 - stop[idx])                              # stop price
        o, lo = O[days], L[days]
        o[:, 0] = entry                                              # day 1 opens at the entry price
        hit = lo <= level[:, None]
        first = np.where(hit.any(1), hit.argmax(1), -1)
        r = C[i + h] / entry - 1
        fired = first >= 0
        k = np.where(fired, first, 0)
        open_k = o[np.arange(len(i)), k]
        gap = fired & (open_k <= level)
        fill = np.where(gap, open_k, level)
        r = np.where(fired, fill / entry - 1, r)
        # worst paper loss until the sale (intraday lows; the sale price if stopped)
        last = np.where(fired, k, h - 1)
        mask = np.arange(h)[None, :] <= last[:, None]
        low_till_exit = np.where(mask, lo, np.inf).min(1)
        w = np.minimum(low_till_exit / entry - 1, r)
        a = amount[idx]
        profit[idx] = a * r
        worst[idx] = np.minimum(a * w, 0)
        stopped[idx], gapped[idx] = fired & (a > 0), gap & (a > 0)
    return pd.DataFrame({"profit": profit, "worst": worst, "stopped": stopped, "gapped": gapped})


def overnight_gap_q95(prices: pd.DataFrame, tickers=None) -> dict[str, float]:
    """For each stock: the 1-in-20 worst overnight drop (open vs the previous close), from
    whatever price table it's given (the training folder when fitting). Plus a pooled default."""
    out, pooled = {}, []
    for t in (tickers if tickers is not None else prices.ticker.unique()):
        px = _adjusted(prices, t)
        if len(px) < 250:
            continue
        drop = (1 - px["open"] / px["close"].shift(1)).clip(lower=0).dropna()
        out[t] = round(float(np.quantile(drop, 0.95)), 5)
        pooled.append(drop)
    out["_default"] = round(float(np.quantile(pd.concat(pooled), 0.95)), 5) if pooled else 0.01
    return out
