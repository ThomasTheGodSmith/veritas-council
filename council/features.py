"""Features and labels for every (date, ticker) row (PLAN.md §1b).

Timing, so nothing looks ahead:
  - A row dated t uses only data up to and including day t's close.
  - The decision is acted on the NEXT trading day: buy at that day's open.
  - The label is the return from that open to the close h trading days after t
    (`fwd_return`), and `y = 1` if it's positive. `mae` is the worst drop from the entry
    price during the holding period, used for the "broke the loss limit" safety metric.
  - `label_known` is the date the label stops being secret (the exit day). A past row may
    only be used as evidence on a decision date strictly after it.
Built from whatever price table it's given, so on the training folder the last h rows of each
ticker simply have no label: the future prices aren't there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H = 20
INDEX = "^DJI"
CROSS_ASSET = {"^VIX": "vix", "^TNX": "tnx", "DX-Y.NYB": "dxy", "CL=F": "oil_futures"}  # oil comes from FRED (longer)

PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "vol_60", "vol_ratio",
    "sma20_sma50", "trend", "rsi", "macd_hist", "z20", "bb_pct", "dist_sma50",
    "dist_high_250", "volume_z",
]
MARKET_FEATURES = [
    "vix", "vix_chg_20", "tnx_chg_20", "dxy_chg_20", "oil_chg_20",
    "dff", "t10y2y", "cpi_yoy", "unrate", "unrate_chg_3m",
]
BREADTH_FEATURES = ["breadth_50", "breadth_chg_20", "ad_chg_20", "divergence"]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _per_ticker(g: pd.DataFrame, h: int) -> pd.DataFrame:
    g = g.sort_values("date")
    c = g["adj_close"]
    r = np.log(c).diff()
    out = pd.DataFrame({"date": g["date"].values, "ticker": g["ticker"].values}, index=g.index)
    for n in (1, 5, 20, 60):
        out[f"ret_{n}"] = c.pct_change(n)
    out["vol_20"] = r.rolling(20).std()
    out["vol_60"] = r.rolling(60).std()
    out["vol_ratio"] = out["vol_20"] / out["vol_60"]
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    sd20 = c.rolling(20).std()
    out["sma20_sma50"] = sma20 / sma50 - 1
    out["trend"] = np.where(sma20 > sma50, 1.0, -1.0)
    out["rsi"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    out["z20"] = (c - sma20) / sd20
    out["bb_pct"] = (c - (sma20 - 2 * sd20)) / (4 * sd20)
    out["dist_sma50"] = c / sma50 - 1
    out["dist_high_250"] = c / c.rolling(250, min_periods=60).max() - 1
    lv = np.log(g["volume"].replace(0, np.nan))
    out["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    out.loc[out["ret_60"].isna(), "trend"] = np.nan

    # labels: buy at the next day's open, sell at the close h days after t
    adj_open = g["open"] * g["adj_close"] / g["close"]
    entry = adj_open.shift(-1)
    exit_ = c.shift(-h)
    out["fwd_return"] = exit_ / entry - 1
    worst = c[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)   # min close over t+1..t+h
    out["mae"] = np.minimum(worst / entry - 1, 0)
    out["y"] = (out["fwd_return"] > 0).astype(float).where(out["fwd_return"].notna())
    out["label_known"] = g["date"].shift(-h)
    out["close"] = g["close"].values
    out["raw_close"] = g["raw_close"].values
    out["adj_close"] = c.values
    return out


def price_features(prices: pd.DataFrame, h: int = H) -> pd.DataFrame:
    stocks = prices[~prices.ticker.isin(CROSS_ASSET)]
    return pd.concat([_per_ticker(g, h) for _, g in stocks.groupby("ticker")], ignore_index=True)


def market_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day of the Dow index: cross-asset moves and macro (as published)."""
    days = prices.loc[prices.ticker == INDEX, ["date"]].sort_values("date").reset_index(drop=True)
    out = days.copy()
    for t, name in CROSS_ASSET.items():
        s = prices.loc[prices.ticker == t, ["date", "adj_close"]].sort_values("date")
        s = pd.merge_asof(days, s, on="date")["adj_close"]
        if name == "vix":
            out["vix"] = s
            out["vix_chg_20"] = s - s.shift(20)
        elif name == "tnx":
            out["tnx_chg_20"] = s - s.shift(20)            # yield change in percentage points
        elif name == "dxy":
            out["dxy_chg_20"] = s.pct_change(20)
    m = macro.pivot_table(index="date", columns="series", values="value").sort_index()
    for sid, col in [("DFF", "dff"), ("T10Y2Y", "t10y2y")]:
        out[col] = pd.merge_asof(days, m[sid].dropna().rename(col).reset_index(), on="date")[col]
    oil = pd.merge_asof(days, m["DCOILWTICO"].dropna().rename("oil").reset_index(), on="date")["oil"]
    out["oil_chg_20"] = oil.pct_change(20)
    cpi = m["CPIAUCSL"].dropna()
    cpi_yoy = (cpi / cpi.shift(12) - 1).rename("cpi_yoy").reset_index()
    out["cpi_yoy"] = pd.merge_asof(days, cpi_yoy, on="date")["cpi_yoy"]
    un = m["UNRATE"].dropna()
    un_df = pd.DataFrame({"date": un.index, "unrate": un.values, "unrate_chg_3m": (un - un.shift(3)).values})
    merged = pd.merge_asof(days, un_df, on="date")
    out["unrate"], out["unrate_chg_3m"] = merged["unrate"], merged["unrate_chg_3m"]
    return out


def member_mask(rows: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """True where rows.ticker was a Dow member on rows.date (point-in-time)."""
    mask = pd.Series(False, index=rows.index)
    for sp in membership.dropna(subset=["ticker"]).itertuples():
        sel = (rows.ticker == sp.ticker) & (rows.date >= sp.start_date)
        if pd.notna(sp.end_date):
            sel &= rows.date < sp.end_date
        mask |= sel
    return mask


def breadth_features(pf: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Per day: share of current members above their 50-day average, advance-decline, divergence.
    Also returns each member's Dow price weight (raw prices, current members with data)."""
    mem = pf[pf.is_member & (pf.ticker != INDEX)].copy()
    mem["above_50"] = (mem["dist_sma50"] > 0).astype(float).where(mem["dist_sma50"].notna())
    mem["adv"] = np.sign(mem["ret_1"])
    daily = mem.groupby("date").agg(breadth_50=("above_50", "mean"), adv=("adv", "sum"), n=("adv", "size"))
    daily["ad_line"] = (daily["adv"] / daily["n"]).cumsum()
    daily["ad_chg_20"] = daily["ad_line"] - daily["ad_line"].shift(20)
    daily["breadth_chg_20"] = daily["breadth_50"] - daily["breadth_50"].shift(20)
    idx = pf[pf.ticker == INDEX].set_index("date")["dist_high_250"]
    daily["divergence"] = ((idx.reindex(daily.index) > -0.02) & (daily["breadth_chg_20"] < -0.10)).astype(float)
    weights = mem["raw_close"] / mem.groupby("date")["raw_close"].transform("sum")
    return daily[BREADTH_FEATURES].reset_index(), weights


def build_panel(prices: pd.DataFrame, macro: pd.DataFrame, membership: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """Every row the models learn from or predict for: the Dow index every day, plus each
    company on the days it was a Dow member. Features, market/breadth context and labels."""
    pf = price_features(prices, h)
    pf["is_member"] = member_mask(pf, membership) | (pf.ticker == INDEX)
    pf = pf[pf.is_member & pf.ticker.ne("DIA")].copy()
    breadth, weights = breadth_features(pf, membership)
    pf["dow_weight"] = weights.reindex(pf.index)
    pf.loc[pf.ticker == INDEX, "dow_weight"] = 1.0
    panel = pf.merge(market_features(prices, macro), on="date", how="left")
    panel = panel.merge(breadth, on="date", how="left")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
