"""Analyst 3, ML Forecaster (PLAN.md §1b-3): gradient-boosted decision trees.

p = sigmoid(f_1(x) + f_2(x) + ... + f_K(x))
  x   = price features (returns over 1/5/20/60 days, volatility, RSI, MACD, z-score, distance
        from the 50-day average and 250-day high, volume change; PRICE_FEATURES in
        council/features.py) + 20-day CHANGES in VIX, the 10-year yield, the dollar and oil.
  f_k = a small decision tree (at most 15 leaves). Each new tree is trained to fix the errors
        the earlier trees still make (LightGBM).
Kept small and regularised on purpose (300 trees, slow learning rate 0.03, every leaf needs
200+ examples, each tree sees 80% of rows and features) so it can't just memorise the
training years. Pooled over all Dow members + the index to have more examples.

Why only changes, not levels (VIX level, Fed funds, CPI, unemployment, yield curve): with the
levels included the trees mostly memorised "which era is this?" (CPI and unemployment were the
top features) and scored 43% on the validation years (2016-2019), below always-buy. With
changes only: 60.7%, Brier 0.237 (best of the feature sets we tried, chosen on validation, never
on test). Economy levels are the Macro analyst's job.

Explanation: LightGBM splits each prediction (in log-odds) into one contribution per feature
(pred_contrib=True, the "SHAP values" of the trees), so top_reasons shows which features pushed
this particular prediction up or down.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from council.analysts.base import Analyst, Context
from council.features import PRICE_FEATURES

SEED = 0


class MLForecaster(Analyst):
    name = "ml_forecaster"
    features = PRICE_FEATURES + ["vix_chg_20", "tnx_chg_20", "dxy_chg_20", "oil_chg_20"]

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        df = panel.dropna(subset=["y"])
        self.model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, n_jobs=4, verbose=-1,
        )
        self.model.fit(df[self.features], df["y"].astype(int))   # LightGBM handles missing values itself
        self._learn_payoffs(df)

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        return self.model.predict_proba(panel[self.features])[:, 1]

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        x = row[self.features].to_frame().T.astype(float)
        contrib = self.model.predict_proba(x, pred_contrib=True)[0][:-1]   # last column is the bias
        order = np.argsort(-np.abs(contrib))
        return [(self.features[i], float(contrib[i])) for i in order]

    def key_numbers(self, row, ctx):
        top = [f for f, _ in self.explain(row, ctx)[:5]]
        return {f: float(row[f]) for f in top if pd.notna(row[f])}


def build() -> MLForecaster:
    return MLForecaster()
