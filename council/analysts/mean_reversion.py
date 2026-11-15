"""Analyst 2, Mean Reversion (PLAN.md §1b-2): "stretched prices tend to snap back".

p = sigmoid(w0 + w1*z + w2*z^2)
  z    = (price - 20-day average) / 20-day standard deviation (how many "standard steps"
         the price is above or below its recent average). Bollinger bands sit at z = +-2.
  z^2  lets the model learn that BIG stretches in either direction behave differently from
       small ones (for example "very stretched either way -> more likely to fall back").
The weights are learned from the training years, so the data decides whether prices really
snap back, rather than us assuming it. Trained on all Dow members + the index together.
(bb_pct is not used: it is exactly (z + 2) / 4, so it would add no new information.)
"""
from __future__ import annotations

import pandas as pd

from council.analysts.base import LogisticAnalyst


class MeanReversion(LogisticAnalyst):
    name = "mean_reversion"
    features = ["z20", "z20_sq"]

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        z = panel["z20"].clip(-5, 5)          # cap rare extreme values so they don't dominate
        return pd.DataFrame({"z20": z, "z20_sq": z ** 2}, index=panel.index)

    def key_numbers(self, row, ctx):
        return {k: float(row[k]) for k in ["z20", "bb_pct", "rsi"] if pd.notna(row[k])}


def build() -> MeanReversion:
    return MeanReversion()
