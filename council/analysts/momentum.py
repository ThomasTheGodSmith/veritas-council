"""Analyst 1, Momentum (PLAN.md §1b): "trends tend to continue".

p = sigmoid(w0 + w1*trend + w2*(RSI-50)/50 + w3*MACD_hist)
  trend      +1 if the 20-day average is above the 50-day average, else -1
  RSI        14-day relative strength index (0-100)
  MACD_hist  MACD minus its signal line, divided by price so every stock is on one scale
Trained on all Dow members + the index together (more examples).
"""
from __future__ import annotations

import pandas as pd

from council.analysts.base import LogisticAnalyst


class Momentum(LogisticAnalyst):
    name = "momentum"
    features = ["trend", "rsi_centred", "macd_hist"]

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "trend": panel["trend"],
            "rsi_centred": (panel["rsi"] - 50) / 50,
            "macd_hist": panel["macd_hist"],
        }, index=panel.index)

    def key_numbers(self, row, ctx):
        return {k: float(row[k]) for k in ["sma20_sma50", "rsi", "macd_hist"] if pd.notna(row[k])}


def build() -> Momentum:
    return Momentum()
