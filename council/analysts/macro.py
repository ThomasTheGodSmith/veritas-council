"""Analyst 9, Macro (PLAN.md §1b-9): "is the economy a headwind or a tailwind?".

p_market = sigmoid(a + b1*fed_funds + b2*(10y - 2y spread) + b3*CPI_yoy
                     + b4*unemployment_change_3m + b5*VIX)
  fed funds rate (dff), yield-curve spread (t10y2y), inflation year-on-year (cpi_yoy),
  3-month change in unemployment (unrate_chg_3m), and the VIX fear index.
  CPI and unemployment are already shifted to their release dates by council/features.py,
  so the model only sees numbers that had actually been published on that day.

Trained ONLY on the Dow index's rows (the question is "will the market go up?").
For a member stock we give the SAME probability: macro numbers are the same for every
company on a given day, so this analyst says "the market-wide wind is blowing this way" and
leaves stock-specific views to the other analysts. This is the simplest honest version;
PLAN.md's beta adjustment (E[r_stock] = beta * E[r_DJIA]) is not used here.
"""
from __future__ import annotations

import pandas as pd

from council.analysts.base import Context, LogisticAnalyst

INDEX = "^DJI"


class Macro(LogisticAnalyst):
    name = "macro"
    features = ["dff", "t10y2y", "cpi_yoy", "unrate_chg_3m", "vix"]
    C = 0.1            # stronger penalty: only ~6,000 index days, and they overlap heavily

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        super().fit(panel[panel.ticker == INDEX], ctx)


def build() -> Macro:
    return Macro()
