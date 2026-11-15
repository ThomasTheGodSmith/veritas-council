"""Drawdown Risk: a trained model of how badly a purchase could go (PLAN.md §9, point 4).

Direction is close to unpredictable, but risk is not: calm stocks tend to stay calm and
turbulent ones turbulent. So instead of "will it go up?", this model answers:

    "In a 1-in-20 bad case, how far below the buying price could this fall
     at some point in the next 20 trading days?"

Target: loss = -mae, the worst drop from the entry price during the holding period (0 if it
never fell below). Model: LightGBM QUANTILE regression at the 95th percentile, so the
prediction is the loss that is exceeded only 5% of the time. That is the same idea as the
Risk Officer's 95% Value-at-Risk, but learned from data instead of assumed from a formula.

The Judge uses it for the hard max-loss rule: buying `amount` passes only if
amount x predicted_loss <= the user's max loss. It doesn't vote on direction.

Trained from scratch on the training years only, pooled across members and the index.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from council.analysts.base import Analyst

FEATURES = [
    "vol_20", "vol_60", "vol_ratio", "ret_1", "ret_5", "ret_20", "ret_60", "dist_high_250",
    "dist_sma50", "z20", "rsi", "volume_z", "vix", "vix_chg_20", "tnx_chg_20", "dxy_chg_20",
    "oil_chg_20", "breadth_50", "breadth_chg_20", "is_index",
]
QUANTILE = 0.95


class DrawdownRisk(Analyst):
    name = "drawdown_risk"
    kind = "risk"            # train.py: predictions are losses, not probabilities of a rise
    features = FEATURES

    def _x(self, panel: pd.DataFrame) -> pd.DataFrame:
        x = panel.reindex(columns=FEATURES).copy()
        x["is_index"] = (panel["ticker"] == "^DJI").astype(float)
        return x.astype(float)

    def fit(self, panel, ctx=None) -> None:
        df = panel.dropna(subset=["mae"])
        loss = (-df["mae"]).clip(lower=0)
        self.model = lgb.LGBMRegressor(
            objective="quantile", alpha=QUANTILE, n_estimators=300, num_leaves=15,
            min_child_samples=200, learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=0, n_jobs=4, verbose=-1)
        self.model.fit(self._x(df), loss)
        self.train_coverage = float((loss > self.predict_loss(df)).mean())

    def predict_loss(self, panel) -> np.ndarray:
        """The 1-in-20 worst drop over the next 20 days, as a fraction (0.08 = 8%)."""
        return np.clip(self.model.predict(self._x(panel)), 0.0, 1.0)

    def predict_proba(self, panel, ctx=None) -> np.ndarray:   # train.py records this column
        return self.predict_loss(panel)

    def explain(self, row, ctx=None):
        contrib = self.model.predict(self._x(row.to_frame().T), pred_contrib=True)[0][:-1]
        order = np.argsort(-np.abs(contrib))
        return [(FEATURES[i], float(contrib[i])) for i in order]

    def key_numbers(self, row, ctx=None):
        return {"predicted_worst_drop_pct": round(float(self.predict_loss(row.to_frame().T)[0]) * 100, 2),
                "vol_20": float(row["vol_20"])}


def build() -> DrawdownRisk:
    return DrawdownRisk()
